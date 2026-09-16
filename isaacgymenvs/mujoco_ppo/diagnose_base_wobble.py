from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from mujoco_ppo.srl_mujoco_env import EnvConfig, SRLMujocoVirEnv


JOINT_NAMES = [
    "left_hip_abd",
    "left_hip_pitch",
    "left_knee",
    "right_hip_abd",
    "right_hip_pitch",
    "right_knee",
]


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


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def peak_to_peak(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(x) - np.min(x))


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


def maybe_write_csv(rows, output_csv):
    if not output_csv:
        return
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved diagnostic trace to:", output_csv)


def summarize(records):
    pitch = wrap_to_pi(records["pitch"])
    roll = wrap_to_pi(records["roll"])
    yaw = wrap_to_pi(records["yaw"])
    wx = np.asarray(records["wx"])
    wy = np.asarray(records["wy"])
    wz = np.asarray(records["wz"])
    vel_x = np.asarray(records["vel_x"])
    root_h = np.asarray(records["root_h"])
    left_contact = np.asarray(records["left_contact"])
    right_contact = np.asarray(records["right_contact"])

    summary = {
        "base_wobble_score": rms(pitch) + rms(roll) + 0.5 * (rms(wx) + rms(wy)),
        "yaw_rms_rad": rms(yaw),
        "pitch_rms_rad": rms(pitch),
        "roll_rms_rad": rms(roll),
        "yaw_pp_rad": peak_to_peak(yaw),
        "pitch_pp_rad": peak_to_peak(pitch),
        "roll_pp_rad": peak_to_peak(roll),
        "wx_rms_rad_s": rms(wx),
        "wy_rms_rad_s": rms(wy),
        "wz_rms_rad_s": rms(wz),
        "vel_x_mean_m_s": float(np.mean(vel_x)),
        "vel_x_std_m_s": float(np.std(vel_x)),
        "root_h_mean_m": float(np.mean(root_h)),
        "root_h_std_m": float(np.std(root_h)),
        "left_contact_ratio": float(np.mean(left_contact)),
        "right_contact_ratio": float(np.mean(right_contact)),
    }
    return summary


def print_joint_diagnostics(records):
    print("\n=== Joint Diagnostics ===")
    abs_roll = np.abs(wrap_to_pi(records["roll"]))
    abs_pitch = np.abs(wrap_to_pi(records["pitch"]))

    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        action = np.asarray(records["action_{}".format(joint_idx)])
        torque = np.asarray(records["torque_{}".format(joint_idx)])
        qvel = np.asarray(records["qvel_{}".format(joint_idx)])
        action_rate = np.diff(action, prepend=action[0])
        torque_rate = np.diff(torque, prepend=torque[0])
        print(
            "{} | action_rms={:.4f} action_rate_rms={:.4f} torque_rms={:.4f} torque_rate_rms={:.4f} qvel_rms={:.4f} corr_roll_torque={:.4f} corr_pitch_torque={:.4f}".format(
                joint_name,
                rms(action),
                rms(action_rate),
                rms(torque),
                rms(torque_rate),
                rms(qvel),
                safe_corr(abs_roll, np.abs(torque)),
                safe_corr(abs_pitch, np.abs(torque)),
            )
        )


def print_event_diagnostics(records):
    print("\n=== Contact / Impact Diagnostics ===")
    left_contact = np.asarray(records["left_contact"], dtype=np.int32)
    right_contact = np.asarray(records["right_contact"], dtype=np.int32)
    wx = np.asarray(records["wx"])
    wy = np.asarray(records["wy"])
    wz = np.asarray(records["wz"])

    left_touchdown = np.sum((left_contact[1:] == 1) & (left_contact[:-1] == 0))
    right_touchdown = np.sum((right_contact[1:] == 1) & (right_contact[:-1] == 0))
    print("left_touchdown_count: {}".format(int(left_touchdown)))
    print("right_touchdown_count: {}".format(int(right_touchdown)))
    print("ang_vel_rms_total: wx={:.4f} wy={:.4f} wz={:.4f}".format(rms(wx), rms(wy), rms(wz)))


def run_diagnosis(args):
    env_cfg = EnvConfig(
        xml_path=args.xml,
        target_vel_x=args.target_vel_x,
        target_ang_vel_z=args.target_ang_vel_z,
        target_height=args.target_height,
    )
    env = SRLMujocoVirEnv(env_cfg)
    policy = create_policy(args.model)

    obs, _ = env.reset(seed=args.seed)

    records = {
        "yaw": [],
        "pitch": [],
        "roll": [],
        "wx": [],
        "wy": [],
        "wz": [],
        "vel_x": [],
        "root_h": [],
        "left_foot_h": [],
        "right_foot_h": [],
        "left_contact": [],
        "right_contact": [],
    }
    for joint_idx in range(env.act_dim):
        records["action_{}".format(joint_idx)] = []
        records["torque_{}".format(joint_idx)] = []
        records["qvel_{}".format(joint_idx)] = []
        records["qpos_{}".format(joint_idx)] = []

    csv_rows = []
    contact_threshold = 0.055

    for step in range(args.steps):
        action = np.clip(policy.predict(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)

        root_rot_mat = env.data.xmat[env.base_id].reshape(3, 3)
        quat = env.data.qpos[3:7]
        euler = quat_to_euler_xyz(quat)
        local_ang_vel = root_rot_mat.T @ env.data.qvel[3:6]
        local_lin_vel = root_rot_mat.T @ env.data.qvel[0:3]
        foot_pos = env._get_srl_end_body_pos()
        left_foot_h = float(foot_pos[0, 2])
        right_foot_h = float(foot_pos[1, 2])
        left_contact = 1 if left_foot_h < contact_threshold else 0
        right_contact = 1 if right_foot_h < contact_threshold else 0

        if step >= args.warmup_steps:
            records["yaw"].append(float(euler[0]))
            records["pitch"].append(float(euler[1]))
            records["roll"].append(float(euler[2]))
            records["wx"].append(float(local_ang_vel[0]))
            records["wy"].append(float(local_ang_vel[1]))
            records["wz"].append(float(local_ang_vel[2]))
            records["vel_x"].append(float(local_lin_vel[0]))
            records["root_h"].append(float(env.data.qpos[2]))
            records["left_foot_h"].append(left_foot_h)
            records["right_foot_h"].append(right_foot_h)
            records["left_contact"].append(left_contact)
            records["right_contact"].append(right_contact)

            row = {
                "step": step,
                "reward": float(reward),
                "root_h": float(env.data.qpos[2]),
                "yaw": float(euler[0]),
                "pitch": float(euler[1]),
                "roll": float(euler[2]),
                "wx": float(local_ang_vel[0]),
                "wy": float(local_ang_vel[1]),
                "wz": float(local_ang_vel[2]),
                "vel_x": float(local_lin_vel[0]),
                "left_foot_h": left_foot_h,
                "right_foot_h": right_foot_h,
                "left_contact": left_contact,
                "right_contact": right_contact,
            }

            for joint_idx in range(env.act_dim):
                records["action_{}".format(joint_idx)].append(float(action[joint_idx]))
                records["torque_{}".format(joint_idx)].append(float(env.last_applied_torques[joint_idx]))
                records["qvel_{}".format(joint_idx)].append(float(env.data.qvel[6 + joint_idx]))
                records["qpos_{}".format(joint_idx)].append(float(env.data.qpos[7 + joint_idx]))
                row["action_{}".format(joint_idx)] = float(action[joint_idx])
                row["torque_{}".format(joint_idx)] = float(env.last_applied_torques[joint_idx])
                row["qvel_{}".format(joint_idx)] = float(env.data.qvel[6 + joint_idx])
                row["qpos_{}".format(joint_idx)] = float(env.data.qpos[7 + joint_idx])

            csv_rows.append(row)

        if args.print_every > 0 and step % args.print_every == 0:
            print(
                "[step {:05d}] reward={:8.4f} root_h={:.3f} yaw={:.3f} pitch={:.3f} roll={:.3f} wx={:.3f} wy={:.3f} wz={:.3f}".format(
                    step,
                    reward,
                    env.data.qpos[2],
                    euler[0],
                    euler[1],
                    euler[2],
                    local_ang_vel[0],
                    local_ang_vel[1],
                    local_ang_vel[2],
                )
            )

        if terminated or truncated:
            print("[episode ended] step={} root_h={:.3f}".format(step, env.data.qpos[2]))
            obs, _ = env.reset()

    summary = summarize(records)
    print("\n=== Base Summary ===")
    for key in sorted(summary.keys()):
        print("{}: {:.6f}".format(key, summary[key]))

    print_joint_diagnostics(records)
    print_event_diagnostics(records)
    if csv_rows:
        maybe_write_csv(csv_rows, args.output_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose where base wobble comes from in MuJoCo.")
    parser.add_argument("--model", type=str, required=True, help="Path to JIT / Isaac checkpoint / finetune checkpoint")
    parser.add_argument("--xml", type=str, default="mjcf/srl_real_v1/srl_real_bot_v1.xml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--target-vel-x", type=float, default=1.0)
    parser.add_argument("--target-ang-vel-z", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-csv", type=str, default="", help="Optional CSV path for per-step trace")
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnosis(parse_args())
