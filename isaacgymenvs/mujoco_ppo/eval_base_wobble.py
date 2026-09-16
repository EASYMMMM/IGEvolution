from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from mujoco_ppo.srl_mujoco_env import EnvConfig, SRLMujocoVirEnv


def quat_to_euler_xyz(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([yaw_z, pitch_y, roll_x], dtype=np.float32)


def wrap_to_pi(x):
    x = np.asarray(x, dtype=np.float64)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class BasePolicy:
    def __init__(self):
        self.mean = np.zeros(153, dtype=np.float32)
        self.std = np.ones(153, dtype=np.float32)

    def predict(self, obs):
        raise NotImplementedError


class JITPolicy(BasePolicy):
    def __init__(self, model_path):
        super().__init__()
        print("Loading JIT policy:", model_path)
        self.actor = torch.jit.load(model_path, map_location="cpu")
        self.actor.eval()

    def predict(self, obs):
        obs_norm = np.clip((obs - self.mean) / self.std, -5.0, 5.0)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.numpy().flatten()


class IsaacCheckpointPolicy(BasePolicy):
    def __init__(self, model_path):
        super().__init__()
        print("Loading Isaac checkpoint policy:", model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_dict = checkpoint["model"]
        self.actor = self._build_actor(model_dict)
        self.actor.eval()

        mean_key = "running_mean_std.running_mean"
        var_key = "running_mean_std.running_var"
        if mean_key in model_dict and var_key in model_dict:
            self.mean = model_dict[mean_key].detach().cpu().numpy().astype(np.float32)
            var = model_dict[var_key].detach().cpu().numpy().astype(np.float32)
            self.std = np.sqrt(var + 1e-8)

    def _build_actor(self, model_dict):
        layers = []
        weights = {}
        biases = {}
        for key, value in model_dict.items():
            if "actor_mlp" not in key:
                continue
            parts = key.split(".")
            idx = int(parts[2])
            if parts[-1] == "weight":
                weights[idx] = value
            elif parts[-1] == "bias":
                biases[idx] = value

        for idx in sorted(weights.keys()):
            weight = weights[idx]
            bias = biases[idx]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            layers.append(torch.nn.ELU())

        mu_weight = model_dict["a2c_network.mu.weight"]
        mu_bias = model_dict["a2c_network.mu.bias"]
        out = torch.nn.Linear(mu_weight.shape[1], mu_weight.shape[0])
        out.weight.data.copy_(mu_weight)
        out.bias.data.copy_(mu_bias)
        layers.append(out)
        return torch.nn.Sequential(*layers)

    def predict(self, obs):
        obs_norm = np.clip((obs - self.mean) / self.std, -5.0, 5.0)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.numpy().flatten()


class FinetunePolicy(BasePolicy):
    def __init__(self, model_path):
        super().__init__()
        print("Loading MuJoCo finetuned policy:", model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        self.actor = self._build_actor(state_dict)
        self.actor.eval()

        mean_key = "obs_norm.running_mean"
        var_key = "obs_norm.running_var"
        if mean_key in state_dict and var_key in state_dict:
            self.mean = state_dict[mean_key].detach().cpu().numpy().astype(np.float32)
            var = state_dict[var_key].detach().cpu().numpy().astype(np.float32)
            self.std = np.sqrt(var + 1e-8)

    def _build_actor(self, state_dict):
        layers = []
        weight_keys = sorted(
            [k for k in state_dict.keys() if k.startswith("actor_mlp.") and k.endswith(".weight")],
            key=lambda x: int(x.split(".")[1]),
        )
        for weight_key in weight_keys:
            idx = int(weight_key.split(".")[1])
            bias_key = "actor_mlp.{}.bias".format(idx)
            weight = state_dict[weight_key]
            bias = state_dict[bias_key]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            layers.append(torch.nn.ELU())

        mu_weight = state_dict["mu.weight"]
        mu_bias = state_dict["mu.bias"]
        out = torch.nn.Linear(mu_weight.shape[1], mu_weight.shape[0])
        out.weight.data.copy_(mu_weight)
        out.bias.data.copy_(mu_bias)
        layers.append(out)
        return torch.nn.Sequential(*layers)

    def predict(self, obs):
        obs_norm = np.clip((obs - self.mean) / self.std, -5.0, 5.0)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.numpy().flatten()


def create_policy(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return FinetunePolicy(model_path)
        if "model" in checkpoint:
            return IsaacCheckpointPolicy(model_path)
    return JITPolicy(model_path)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def peak_to_peak(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(x) - np.min(x))


def summarize_metrics(series):
    yaw = wrap_to_pi(series["yaw"])
    pitch = wrap_to_pi(series["pitch"])
    roll = wrap_to_pi(series["roll"])
    wx = np.asarray(series["wx"])
    wy = np.asarray(series["wy"])
    wz = np.asarray(series["wz"])
    root_h = np.asarray(series["root_h"])
    vel_x = np.asarray(series["vel_x"])

    metrics = {
        "yaw_rms_rad": rms(yaw),
        "pitch_rms_rad": rms(pitch),
        "roll_rms_rad": rms(roll),
        "yaw_pp_rad": peak_to_peak(yaw),
        "pitch_pp_rad": peak_to_peak(pitch),
        "roll_pp_rad": peak_to_peak(roll),
        "wx_rms_rad_s": rms(wx),
        "wy_rms_rad_s": rms(wy),
        "wz_rms_rad_s": rms(wz),
        "root_h_mean_m": float(np.mean(root_h)),
        "root_h_std_m": float(np.std(root_h)),
        "vel_x_mean_m_s": float(np.mean(vel_x)),
        "vel_x_std_m_s": float(np.std(vel_x)),
    }
    metrics["base_wobble_score"] = (
        metrics["pitch_rms_rad"]
        + metrics["roll_rms_rad"]
        + 0.5 * (metrics["wy_rms_rad_s"] + metrics["wx_rms_rad_s"])
    )
    return metrics


def run_eval(args):
    env_cfg = EnvConfig(
        xml_path=args.xml,
        target_vel_x=args.target_vel_x,
        target_ang_vel_z=args.target_ang_vel_z,
        target_height=args.target_height,
    )
    if args.disable_proxy:
        env_cfg.proxy_fx_gain = 0.0
        env_cfg.proxy_fz_bias = 0.0
        env_cfg.proxy_kx = 0.0
        env_cfg.proxy_cx = 0.0
        env_cfg.proxy_kz = 0.0
        env_cfg.proxy_cz = 0.0
        env_cfg.proxy_kt = 0.0
        env_cfg.proxy_ct = 0.0
    env = SRLMujocoVirEnv(env_cfg)
    policy = create_policy(args.model)

    obs, _ = env.reset(seed=args.seed)

    series = {
        "yaw": [],
        "pitch": [],
        "roll": [],
        "wx": [],
        "wy": [],
        "wz": [],
        "root_h": [],
        "vel_x": [],
    }

    for step in range(args.steps):
        action = policy.predict(obs)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)

        root_rot_mat = env.data.xmat[env.base_id].reshape(3, 3)
        quat = env.data.qpos[3:7]
        euler = quat_to_euler_xyz(quat)
        local_ang_vel = root_rot_mat.T @ env.data.qvel[3:6]
        local_lin_vel = root_rot_mat.T @ env.data.qvel[0:3]

        if step >= args.warmup_steps:
            series["yaw"].append(float(euler[0]))
            series["pitch"].append(float(euler[1]))
            series["roll"].append(float(euler[2]))
            series["wx"].append(float(local_ang_vel[0]))
            series["wy"].append(float(local_ang_vel[1]))
            series["wz"].append(float(local_ang_vel[2]))
            series["root_h"].append(float(env.data.qpos[2]))
            series["vel_x"].append(float(local_lin_vel[0]))

        if args.print_every > 0 and step % args.print_every == 0:
            print(
                "[step {:05d}] reward={:8.4f} root_h={:.3f} yaw={:.3f} pitch={:.3f} roll={:.3f}".format(
                    step,
                    reward,
                    env.data.qpos[2],
                    euler[0],
                    euler[1],
                    euler[2],
                )
            )

        if terminated or truncated:
            print("[episode ended] step={} root_h={:.3f}".format(step, env.data.qpos[2]))
            obs, _ = env.reset()

    metrics = summarize_metrics(series)
    print("\n=== Base Wobble Metrics ===")
    for key in sorted(metrics.keys()):
        print("{}: {:.6f}".format(key, metrics[key]))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base wobble metrics in MuJoCo.")
    parser.add_argument("--model", type=str, required=True, help="Path to JIT / Isaac checkpoint / finetune checkpoint")
    parser.add_argument("--xml", type=str, default="mjcf/srl_real_v1/srl_real_bot_v1.xml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--target-vel-x", type=float, default=1.0)
    parser.add_argument("--target-ang-vel-z", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=1.0)
    parser.add_argument("--disable-proxy", action="store_true", help="Disable virtual human interaction forces during evaluation")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
