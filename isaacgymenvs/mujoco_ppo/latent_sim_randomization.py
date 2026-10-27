"""Opt-in Isaac-range randomization for the standalone latent MuJoCo viewer."""

from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json

import mujoco
import numpy as np
import yaml


@dataclass
class LatentDRConfig:
    action_delay_ms: tuple = (0.0, 5.0, 10.0)
    armature_range: tuple = (0.8, 1.2)
    inertia_range: tuple = (0.8, 1.2)
    mass_range: tuple = (0.5, 1.5)
    kp_range: tuple = (0.8, 1.2)
    kd_range: tuple = (0.8, 1.2)
    friction_range: tuple = (0.7, 1.3)
    gravity_std: float = 0.4
    joint_limit_std: float = 0.01
    observation_noise_std: float = 0.002
    action_noise_std: float = 0.02
    velocity_perturbation_probability: float = 0.002
    velocity_perturbation_range: tuple = (0.2, 0.8)
    runtime_resample_steps: int = 6000
    sensor_noise_mode: str = "legacy"
    sensor_noise_orientation_max: float = 0.05
    sensor_noise_ang_vel_max: float = 0.2
    sensor_noise_dof_pos_max: float = 0.01
    sensor_noise_dof_vel_max: float = 0.5

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("DR config must contain a YAML mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown DR fields: {}".format(sorted(unknown)))
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.sensor_noise_mode not in ("legacy", "gaussian", "uniform", "none"):
            raise ValueError("Invalid sensor_noise_mode")
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name.endswith("_range"):
                pair = np.asarray(value, dtype=float)
                if (pair.shape != (2,) or not np.isfinite(pair).all()
                        or pair[0] > pair[1] or pair[0] < 0
                        or (name != "velocity_perturbation_range" and pair[0] == 0)):
                    raise ValueError("Invalid {}".format(name))
            elif name.endswith("_std"):
                if not np.isfinite(value) or value < 0:
                    raise ValueError("{} must be finite and nonnegative".format(name))
            elif name.startswith("sensor_noise_") and name.endswith("_max"):
                if not np.isfinite(value) or value < 0:
                    raise ValueError("{} must be finite and nonnegative".format(name))
        if (not np.isfinite(self.velocity_perturbation_probability)
                or not 0 <= self.velocity_perturbation_probability <= 1):
            raise ValueError("Velocity perturbation probability must be in [0, 1]")
        if (not isinstance(self.runtime_resample_steps, int)
                or self.runtime_resample_steps < 1):
            raise ValueError("runtime_resample_steps must be a positive integer")
        delay_steps(self.action_delay_ms, 0.005)


def delay_steps(choices_ms, physics_dt):
    choices = np.asarray(choices_ms, dtype=float)
    if (choices.ndim != 1 or not choices.size or not np.isfinite(choices).all()
            or np.any(choices < 0) or not np.isfinite(physics_dt) or physics_dt <= 0):
        raise ValueError("Invalid delay choices or physics timestep")
    steps = choices / (1000 * physics_dt)
    if not np.allclose(steps, np.rint(steps), atol=1e-8, rtol=0):
        raise ValueError("Delay must be a multiple of the 5 ms physics/control tick")
    if np.unique(choices).size != choices.size:
        raise ValueError("Delay choices must be unique")
    return np.rint(steps).astype(int)


class PhysicsTargetDelay:
    def __init__(self, initial_target, steps):
        if int(steps) != steps or steps < 0:
            raise ValueError("Delay steps must be a nonnegative integer")
        self.queue = deque(np.asarray(initial_target).copy() for _ in range(int(steps)))

    def advance(self, target):
        self.queue.append(np.asarray(target).copy())
        return self.queue.popleft()


class LatentSimulationDR:
    def __init__(self, model, joints, dofs, base_id, kp, kd, seed, enabled,
                 config, resample_each_episode=False, velocity_enabled=True):
        config.validate()
        self.model, self.joints, self.dofs = model, joints, dofs
        self.enabled, self.config = bool(enabled), config
        self.resample_each_episode = resample_each_episode
        self.velocity_enabled = velocity_enabled
        streams = np.random.SeedSequence(seed).spawn(5)
        self.parameter_rng, self.delay_rng, self.noise_rng, self.velocity_rng = (
            np.random.default_rng(stream) for stream in streams[:4]
        )
        self.sensor_noise_rng = np.random.default_rng(streams[4])
        self.body_ids = np.array([
            index for index in range(1, model.nbody)
            if self._descends_from(index, base_id)
        ], dtype=int)
        self.geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, self.body_ids)
                                      & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        self.nominal = {
            "mass": model.body_mass[self.body_ids].copy(),
            "inertia": model.body_inertia[self.body_ids].copy(),
            "armature": model.dof_armature[dofs].copy(),
            "friction": model.geom_friction[self.geom_ids].copy(),
            "joint_range": model.jnt_range[joints].copy(),
            "gravity": model.opt.gravity.copy(),
        }
        if enabled and np.any(self.nominal["armature"] <= 0):
            raise ValueError("Armature DR requires positive XML joint armature")
        self.nominal_kp, self.nominal_kd = kp.copy(), kd.copy()
        self.kp, self.kd = kp.copy(), kd.copy()
        self.delay_choices = delay_steps(config.action_delay_ms, 0.005)
        self.total_steps = 0
        self.last_runtime_sample_step = None
        self.setup_sample = None
        self.runtime_sample = None

    def _descends_from(self, body_id, ancestor):
        while body_id:
            if body_id == ancestor:
                return True
            body_id = self.model.body_parentid[body_id]
        return False

    def reset(self, data, initial_target):
        if self.enabled:
            cfg, rng, model = self.config, self.parameter_rng, self.model
            setup_new = self.setup_sample is None or self.resample_each_episode
            if setup_new:
                self.setup_sample = {
                    "mass": rng.uniform(*cfg.mass_range, len(self.body_ids)),
                    "inertia": rng.uniform(*cfg.inertia_range, len(self.body_ids)),
                    "armature": rng.uniform(*cfg.armature_range, len(self.dofs)),
                }
            runtime_new = (self.runtime_sample is None or self.resample_each_episode
                           or self.total_steps - self.last_runtime_sample_step >= cfg.runtime_resample_steps)
            if runtime_new:
                self.runtime_sample = {
                    "kp": rng.uniform(*cfg.kp_range, len(self.dofs)),
                    "kd": rng.uniform(*cfg.kd_range, len(self.dofs)),
                    "friction": rng.uniform(*cfg.friction_range, len(self.geom_ids)),
                    "gravity": rng.normal(0, cfg.gravity_std, 3),
                    "joint_range": rng.normal(0, cfg.joint_limit_std, (len(self.joints), 2)),
                }
                self.last_runtime_sample_step = self.total_steps
            setup, runtime, nominal = self.setup_sample, self.runtime_sample, self.nominal
            model.body_mass[self.body_ids] = nominal["mass"] * setup["mass"]
            # Mass and inertia are independent MuJoCo properties; do not move CoM.
            model.body_inertia[self.body_ids] = nominal["inertia"] * setup["inertia"][:, None]
            model.dof_armature[self.dofs] = nominal["armature"] * setup["armature"]
            model.geom_friction[self.geom_ids] = nominal["friction"]
            model.geom_friction[self.geom_ids, 0] *= runtime["friction"]
            model.jnt_range[self.joints] = nominal["joint_range"] + runtime["joint_range"]
            if np.any(model.jnt_range[self.joints, 0] >= model.jnt_range[self.joints, 1]):
                raise ValueError("Randomized joint limits are inverted")
            model.opt.gravity[:] = nominal["gravity"] + runtime["gravity"]
            self.kp = self.nominal_kp * runtime["kp"]
            self.kd = self.nominal_kd * runtime["kd"]
            mujoco.mj_setConst(model, data)
            self.delay_ms = float(self.delay_rng.choice(cfg.action_delay_ms))
        else:
            self.delay_ms = 0.0
        self.delay = PhysicsTargetDelay(initial_target, int(round(self.delay_ms / 5)))
        return self.parameters()

    def parameters(self):
        model = self.model
        return {
            "enabled": self.enabled, "delay_ms": self.delay_ms,
            "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i))
                           for i in self.body_ids],
            "body_mass": model.body_mass[self.body_ids].tolist(),
            "body_inertia": model.body_inertia[self.body_ids].tolist(),
            "armature": model.dof_armature[self.dofs].tolist(),
            "geom_ids": self.geom_ids.tolist(),
            "geom_friction": model.geom_friction[self.geom_ids].tolist(),
            "joint_range": model.jnt_range[self.joints].tolist(),
            "gravity": model.opt.gravity.tolist(),
            "kp": self.kp.tolist(), "kd": self.kd.tolist(),
            "setup_multipliers": None if self.setup_sample is None else {
                key: value.tolist() for key, value in self.setup_sample.items()
            },
            "runtime_samples": None if self.runtime_sample is None else {
                key: value.tolist() for key, value in self.runtime_sample.items()
            },
        }

    def observation_noise(self):
        if not self.enabled:
            return None
        noise = self.noise_rng.normal(0, self.config.observation_noise_std, 133).astype(np.float32)
        # Keep the action-noise RNG sequence paired with legacy evaluation.
        return noise if self.config.sensor_noise_mode in ("legacy", "gaussian") else None

    def sensor_observation_noise(self, dof_vel_scale=0.05):
        if not self.enabled or self.config.sensor_noise_mode != "uniform":
            return None
        cfg = self.config
        amplitudes = np.array([cfg.sensor_noise_ang_vel_max] * 3
                              + [cfg.sensor_noise_orientation_max] * 3
                              + [cfg.sensor_noise_dof_pos_max] * 6
                              + [cfg.sensor_noise_dof_vel_max] * 6, dtype=np.float32)
        # Same physical bounds as Isaac, followed by the joint-velocity scale.
        noise = np.zeros(26, dtype=np.float32)
        noise[:18] = (self.sensor_noise_rng.uniform(-1, 1, 18) * amplitudes).astype(np.float32)
        noise[12:18] *= dof_vel_scale
        return noise

    def action(self, action):
        if not self.enabled:
            return action.copy()
        return np.clip(action + self.noise_rng.normal(0, self.config.action_noise_std, 6), -1, 1)

    def velocity_event(self, data, step, timing):
        if not self.enabled:
            return None
        rng, cfg = self.velocity_rng, self.config
        if rng.random() >= cfg.velocity_perturbation_probability:
            return None
        magnitude = float(rng.uniform(*cfg.velocity_perturbation_range))
        angle = float(rng.uniform(0, 2 * np.pi))
        delta = magnitude * np.array([np.cos(angle), np.sin(angle)])
        before = data.qvel[:2].copy()
        if self.velocity_enabled:
            data.qvel[:2] += delta
            mujoco.mj_forward(self.model, data)
        return {
            "step": step, "time_s": float(data.time), "timing": timing,
            "applied": bool(self.velocity_enabled), "magnitude": magnitude,
            "delta_v_world_xy": delta.tolist(), "before_world_xy": before.tolist(),
            "after_world_xy": data.qvel[:2].tolist(),
        }


class SimulationJournal:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        with (self.directory / "settings.json").open("w", encoding="utf-8") as stream:
            json.dump(settings, stream, indent=2, allow_nan=False)
        self.stream = (self.directory / "events.jsonl").open("w", encoding="utf-8")

    def write(self, event_type, episode, values):
        self.stream.write(json.dumps({"type": event_type, "episode": episode, **values},
                                     allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()
