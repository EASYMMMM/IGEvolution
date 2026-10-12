"""Opt-in SRL dynamics variation without changing policy observations."""

import numpy as np
import torch


def positive_range(values, name):
    bounds = np.asarray(values, dtype=np.float64)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[0] <= 0.0 or bounds[1] < bounds[0]):
        raise ValueError("{} must be two finite, positive, ordered values".format(name))
    return tuple(bounds)


def delay_steps_from_ms(values, physics_dt):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or np.any(values < 0.0) or not np.isfinite(physics_dt)
            or physics_dt <= 0.0):
        raise ValueError("Action delays and physics dt must be finite and valid")
    steps = values / (1000.0 * physics_dt)
    rounded = np.rint(steps)
    if not np.allclose(steps, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("Action delays must be multiples of sim.dt, not the policy dt")
    if len(np.unique(rounded)) != len(rounded):
        raise ValueError("Action delay choices must be distinct")
    return rounded.astype(np.int64)


class PhysicsTargetDelay:
    """Per-environment delays for targets held between policy updates."""

    def __init__(self, initial_targets, choices):
        self.choices = torch.as_tensor(
            choices, device=initial_targets.device, dtype=torch.long
        )
        self.capacity = int(max(choices)) + 1
        self.history = initial_targets.unsqueeze(0).repeat(self.capacity, 1, 1)
        self.env_indices = torch.arange(initial_targets.shape[0], device=initial_targets.device)
        self.steps = torch.zeros_like(self.env_indices)
        self.cursor = 0
        self.reset(self.env_indices, initial_targets)

    def reset(self, env_ids, initial_targets):
        self.history[:, env_ids, :] = initial_targets.unsqueeze(0)
        draws = torch.randint(len(self.choices), (len(env_ids),), device=self.steps.device)
        self.steps[env_ids] = self.choices[draws]

    def advance(self, targets):
        self.history[self.cursor].copy_(targets)
        read_slots = (self.cursor - self.steps) % self.capacity
        delayed = self.history[read_slots, self.env_indices]
        self.cursor = (self.cursor + 1) % self.capacity
        return delayed


def inertia_matrix(value):
    return np.asarray([
        [getattr(value, row).x, getattr(value, row).y, getattr(value, row).z]
        for row in ("x", "y", "z")
    ], dtype=np.float64)


def _write_matrix(value, matrix):
    for index, name in enumerate(("x", "y", "z")):
        row = getattr(value, name)
        row.x, row.y, row.z = map(float, matrix[index])
        setattr(value, name, row)
    return value


def scale_body_inertia(body, factor):
    matrix = inertia_matrix(body.inertia)
    if (not np.isfinite(factor) or factor <= 0.0
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-9)):
        raise ValueError("Body inertia and multiplier must be finite and physical")
    principal = np.linalg.eigvalsh(matrix)
    if (np.any(principal <= 0.0)
            or principal[-1] > principal[0] + principal[1] + 1e-6 * principal[-1]):
        raise ValueError("Body inertia must be positive and satisfy the triangle inequality")
    scaled = factor * matrix
    body.inertia = _write_matrix(body.inertia, scaled)
    if hasattr(body, "invInertia"):
        body.invInertia = _write_matrix(body.invInertia, np.linalg.inv(scaled))
    return scaled


class SRLDynamicsRandomizationMixin:
    """Setup-only armature/inertia variation and episode-wise target latency.

    Feature flags are independent of the legacy task.task.randomize switch.
    Body properties are set after legacy startup DR, before prepare_sim().
    """

    def __init__(self, cfg, *args, **kwargs):
        options = cfg["env"].get("srl_dynamics_dr", {})
        enabled = bool(options.get("enable", False))
        self._srl_delay_enabled = enabled and bool(options.get("action_delay_enable", True))
        self._srl_armature_enabled = enabled and bool(options.get("armature_enable", True))
        self._srl_inertia_enabled = enabled and bool(options.get("inertia_enable", True))
        self._srl_armature_range = positive_range(
            options.get("armature_range", (0.8, 1.2)), "armature_range"
        )
        self._srl_inertia_range = positive_range(
            options.get("inertia_range", (0.8, 1.2)), "inertia_range"
        )
        self._srl_armature_nominal = options.get("armature_nominal", None)
        self._srl_delay_choices = delay_steps_from_ms(
            options.get("action_delay_ms", (0.0, 5.0, 10.0)), float(cfg["sim"]["dt"])
        ) if self._srl_delay_enabled else None
        if self._srl_delay_enabled and not (
            cfg["env"].get("pdControl", True) or cfg["env"].get("forceControl", False)
        ):
            raise ValueError("SRL target delay requires PD position or torque control")
        self._srl_target_delay = None
        super().__init__(cfg, *args, **kwargs)
        self.applied_pd_targets = self.filtered_pd_targets.clone()
        if enabled:
            print("SRL action scale: {} (fixed)".format(self.action_scale))
        if self._srl_delay_enabled:
            self._srl_target_delay = PhysicsTargetDelay(self.dof_pos, self._srl_delay_choices)
            print("SRL target delay: choices={} ms, sampled once per environment episode".format(
                (self._srl_delay_choices * float(cfg["sim"]["dt"]) * 1000.0).tolist()
            ))

    def _configure_srl_asset_options(self, options):
        super()._configure_srl_asset_options(options)
        if self._srl_armature_enabled:
            # Use joint-space armature, not an addition to link inertia tensors.
            options.use_physx_armature = True

    def create_sim(self):
        super().create_sim()
        if self._srl_armature_enabled or self._srl_inertia_enabled:
            self._setup_srl_physical_variation()

    def _setup_srl_physical_variation(self):
        armature_scales = np.random.uniform(
            *self._srl_armature_range, size=(self.num_envs, self.num_dof)
        ) if self._srl_armature_enabled else None
        inertia_scales = np.random.uniform(
            *self._srl_inertia_range, size=(self.num_envs, self.num_bodies)
        ) if self._srl_inertia_enabled else None
        nominal_override = None
        if self._srl_armature_enabled and self._srl_armature_nominal is not None:
            nominal_override = np.asarray(self._srl_armature_nominal, dtype=np.float64)
            if (nominal_override.shape != (self.num_dof,)
                    or not np.isfinite(nominal_override).all()
                    or np.any(nominal_override <= 0.0)):
                raise ValueError("armature_nominal must contain one positive value per DOF")

        for index, (env, actor) in enumerate(zip(self.envs, self.humanoid_handles)):
            if self._srl_armature_enabled:
                props = self.gym.get_actor_dof_properties(env, actor)
                if "armature" not in props.dtype.names:
                    raise RuntimeError("This Isaac Gym version has no DOF armature property")
                nominal = (nominal_override if nominal_override is not None
                           else props["armature"].astype(np.float64).copy())
                if not np.isfinite(nominal).all() or np.any(nominal <= 0.0):
                    raise ValueError(
                        "Imported joint armature is zero or invalid; check the XML and "
                        "use_physx_armature, or specify srl_dynamics_dr.armature_nominal"
                    )
                if index == 0:
                    print("SRL nominal joint armature: {}".format(nominal.tolist()))
                props["armature"] = nominal * armature_scales[index]
                self.gym.set_actor_dof_properties(env, actor, props)
                actual = self.gym.get_actor_dof_properties(env, actor)["armature"]
                if not np.allclose(actual, props["armature"], rtol=1e-5, atol=1e-8):
                    raise RuntimeError("Isaac Gym did not retain the randomized DOF armature")

            if self._srl_inertia_enabled:
                bodies = self.gym.get_actor_rigid_body_properties(env, actor)
                masses = np.asarray([body.mass for body in bodies])
                expected = [
                    scale_body_inertia(body, factor)
                    for body, factor in zip(bodies, inertia_scales[index])
                ]
                self.gym.set_actor_rigid_body_properties(env, actor, bodies, False)
                actual = self.gym.get_actor_rigid_body_properties(env, actor)
                if not np.allclose([body.mass for body in actual], masses):
                    raise RuntimeError("Inertia randomization unexpectedly changed body masses")
                for body, wanted in zip(actual, expected):
                    if not np.allclose(inertia_matrix(body.inertia), wanted, rtol=1e-4, atol=1e-8):
                        raise RuntimeError("Isaac Gym did not retain the randomized link inertia")

        self.srl_armature_scales = armature_scales
        self.srl_inertia_scales = inertia_scales
        for name, scales in (("armature", armature_scales), ("link inertia", inertia_scales)):
            if scales is not None:
                print("SRL {}: setup-only independent multipliers [{:.4f}, {:.4f}] (readback verified)".format(
                    name, float(scales.min()), float(scales.max())
                ))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._srl_target_delay is not None:
            env_ids = env_ids.reshape(-1).to(device=self.device, dtype=torch.long)
            self._srl_target_delay.reset(env_ids, self.dof_pos[env_ids])
            self.applied_pd_targets[env_ids] = self.dof_pos[env_ids]

    def _apply_srl_pd_targets(self, targets):
        if self._srl_target_delay is None:
            super()._apply_srl_pd_targets(targets)
        # With latency, submit only from the physics-step hook below.

    def pre_simulation_step(self, substep_index):
        super().pre_simulation_step(substep_index)
        if self._srl_target_delay is None:
            return
        self.applied_pd_targets = self._srl_target_delay.advance(self.filtered_pd_targets)
        if self._force_control:
            if self.device == "cpu" and substep_index > 0:
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        super()._apply_srl_pd_targets(self.applied_pd_targets)
