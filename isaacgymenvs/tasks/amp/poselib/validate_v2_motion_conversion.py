"""Validate legacy AMP motion -> v2 XYZ hinge-chain conversion.

The core checks only require NumPy and the Python standard library. Matplotlib
plots and Isaac Gym / MotionLibV2 checks are optional, so the data-level
validation can still run in lightweight environments.

This script never modifies either input motion file.
"""

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import numpy as np


JOINT_GROUPS = OrderedDict([
    ("abdomen", ("torso", ("abdomen_x_link", "abdomen_y_link", "torso"))),
    ("neck", ("head", ("neck_x_link", "neck_y_link", "head"))),
    (
        "right_shoulder",
        (
            "right_upper_arm",
            ("right_shoulder_x_link", "right_shoulder_y_link", "right_upper_arm"),
        ),
    ),
    (
        "left_shoulder",
        (
            "left_upper_arm",
            ("left_shoulder_x_link", "left_shoulder_y_link", "left_upper_arm"),
        ),
    ),
    ("right_hip", ("right_thigh", ("right_hip_x_link", "right_hip_y_link", "right_thigh"))),
    (
        "right_ankle",
        ("right_foot", ("right_ankle_x_link", "right_ankle_y_link", "right_foot")),
    ),
    ("left_hip", ("left_thigh", ("left_hip_x_link", "left_hip_y_link", "left_thigh"))),
    (
        "left_ankle",
        ("left_foot", ("left_ankle_x_link", "left_ankle_y_link", "left_foot")),
    ),
])

SCALAR_JOINTS = OrderedDict([
    ("right_elbow", ("right_lower_arm", 1)),
    ("left_elbow", ("left_lower_arm", 1)),
    ("right_knee", ("right_shin", 1)),
    ("left_knee", ("left_shin", 1)),
])

REAL_BODIES = [
    "pelvis",
    "torso",
    "head",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
]

INSERTED_NODES = [
    "abdomen_x_link",
    "abdomen_y_link",
    "neck_x_link",
    "neck_y_link",
    "right_shoulder_x_link",
    "right_shoulder_y_link",
    "left_shoulder_x_link",
    "left_shoulder_y_link",
    "right_hip_x_link",
    "right_hip_y_link",
    "left_hip_x_link",
    "left_hip_y_link",
    "right_ankle_x_link",
    "right_ankle_y_link",
    "left_ankle_x_link",
    "left_ankle_y_link",
]

# This is the 28-D policy / Isaac Gym ordering used by the humanoid task.
DOF_SPECS = [
    ("abdomen_x", "abdomen_x_link", 0),
    ("abdomen_y", "abdomen_y_link", 1),
    ("abdomen_z", "torso", 2),
    ("neck_x", "neck_x_link", 0),
    ("neck_y", "neck_y_link", 1),
    ("neck_z", "head", 2),
    ("right_shoulder_x", "right_shoulder_x_link", 0),
    ("right_shoulder_y", "right_shoulder_y_link", 1),
    ("right_shoulder_z", "right_upper_arm", 2),
    ("right_elbow", "right_lower_arm", 1),
    ("left_shoulder_x", "left_shoulder_x_link", 0),
    ("left_shoulder_y", "left_shoulder_y_link", 1),
    ("left_shoulder_z", "left_upper_arm", 2),
    ("left_elbow", "left_lower_arm", 1),
    ("right_hip_x", "right_hip_x_link", 0),
    ("right_hip_y", "right_hip_y_link", 1),
    ("right_hip_z", "right_thigh", 2),
    ("right_knee", "right_shin", 1),
    ("right_ankle_x", "right_ankle_x_link", 0),
    ("right_ankle_y", "right_ankle_y_link", 1),
    ("right_ankle_z", "right_foot", 2),
    ("left_hip_x", "left_hip_x_link", 0),
    ("left_hip_y", "left_hip_y_link", 1),
    ("left_hip_z", "left_thigh", 2),
    ("left_knee", "left_shin", 1),
    ("left_ankle_x", "left_ankle_x_link", 0),
    ("left_ankle_y", "left_ankle_y_link", 1),
    ("left_ankle_z", "left_foot", 2),
]

DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]


class ValidationError(RuntimeError):
    pass


class MotionData(object):
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = np.load(str(self.path), allow_pickle=True)
        if raw.shape != () or raw.dtype != object:
            raise ValidationError("{} is not a Poselib object npy".format(self.path))
        data = raw.item()

        self.rotation = _tensor(data["rotation"]).astype(np.float64)
        self.root_translation = _tensor(data["root_translation"]).astype(np.float64)
        self.fps = float(np.asarray(data.get("fps", 0.0)))
        self.is_local = bool(data.get("is_local", True))

        tree = data["skeleton_tree"]
        self.node_names = [str(x) for x in np.asarray(tree["node_names"]).tolist()]
        self.parent_indices = _tensor(tree["parent_indices"]).astype(np.int64)
        self.local_translation = _tensor(tree["local_translation"]).astype(np.float64)
        self.node_index = {name: i for i, name in enumerate(self.node_names)}

        if not self.is_local:
            raise ValidationError("{} stores global rotations; local rotations are required".format(self.path))
        if self.rotation.ndim != 3 or self.rotation.shape[-1] != 4:
            raise ValidationError("Unexpected local rotation shape: {}".format(self.rotation.shape))
        if self.rotation.shape[1] != len(self.node_names):
            raise ValidationError("Rotation node count does not match skeleton node count")
        if self.root_translation.shape != (self.rotation.shape[0], 3):
            raise ValidationError("Unexpected root translation shape: {}".format(self.root_translation.shape))
        if self.parent_indices.shape != (len(self.node_names),):
            raise ValidationError("Unexpected parent index shape")
        if self.local_translation.shape != (len(self.node_names), 3):
            raise ValidationError("Unexpected local translation shape")
        if self.fps <= 0.0:
            raise ValidationError("Motion FPS must be positive")

    @property
    def frames(self):
        return self.rotation.shape[0]

    @property
    def duration(self):
        return (self.frames - 1) / self.fps


class TposeData(object):
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = np.load(str(self.path), allow_pickle=True)
        data = raw.item()
        self.rotation = _tensor(data["rotation"]).astype(np.float64)
        if self.rotation.ndim == 3 and self.rotation.shape[0] == 1:
            self.rotation = self.rotation[0]
        tree = data["skeleton_tree"]
        self.node_names = [str(x) for x in np.asarray(tree["node_names"]).tolist()]
        self.node_index = {name: i for i, name in enumerate(self.node_names)}
        if self.rotation.shape != (len(self.node_names), 4):
            raise ValidationError("Unexpected T-pose rotation shape: {}".format(self.rotation.shape))


def _tensor(value):
    if isinstance(value, dict) and "arr" in value:
        return np.asarray(value["arr"])
    return np.asarray(value)


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValidationError("Zero-length quaternion encountered")
    return q / norm


def quat_mul(a, b):
    """Hamilton product for XYZW quaternions; independent of project helpers."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_conjugate(q):
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :3] *= -1.0
    return out


def quat_rotate(q, v):
    q = normalize_quat(q)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., :3]
    qw = q[..., 3:4]
    return v + 2.0 * (qw * np.cross(qv, v) + np.cross(qv, np.cross(qv, v)))


def quat_distance(a, b):
    dot = np.sum(normalize_quat(a) * normalize_quat(b), axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def quat_to_matrix(q):
    q = normalize_quat(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def axis_quat(angle, axis):
    angle = np.asarray(angle, dtype=np.float64)
    out = np.zeros(angle.shape + (4,), dtype=np.float64)
    out[..., axis] = np.sin(0.5 * angle)
    out[..., 3] = np.cos(0.5 * angle)
    return out


def axis_matrix(angle, axis):
    angle = np.asarray(angle, dtype=np.float64)
    c = np.cos(angle)
    s = np.sin(angle)
    out = np.zeros(angle.shape + (3, 3), dtype=np.float64)
    if axis == 0:
        out[..., 0, 0] = 1
        out[..., 1, 1] = c
        out[..., 1, 2] = -s
        out[..., 2, 1] = s
        out[..., 2, 2] = c
    elif axis == 1:
        out[..., 0, 0] = c
        out[..., 0, 2] = s
        out[..., 1, 1] = 1
        out[..., 2, 0] = -s
        out[..., 2, 2] = c
    else:
        out[..., 0, 0] = c
        out[..., 0, 1] = -s
        out[..., 1, 0] = s
        out[..., 1, 1] = c
        out[..., 2, 2] = 1
    return out


def single_axis_angle(q, axis):
    """Extract signed angle from a quaternion expected to be pure axis rotation."""
    q = normalize_quat(q).copy()
    q *= np.where(q[..., 3:4] < 0.0, -1.0, 1.0)
    return wrap_to_pi(2.0 * np.arctan2(q[..., axis], q[..., 3]))


def quat_to_hinge_xyz_chain(q):
    """Decompose Qx(qx) * Qy(qy) * Qz(qz) into nested hinge angles."""
    q = normalize_quat(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.arctan2(2.0 * (w * x - y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (x * z + w * y), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        axis=-1,
    )


def wrap_to_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def obs6(q):
    shape = q.shape[:-1] + (3,)
    tangent = np.zeros(shape, dtype=np.float64)
    normal = np.zeros(shape, dtype=np.float64)
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return np.concatenate([quat_rotate(q, tangent), quat_rotate(q, normal)], axis=-1)


def forward_kinematics(motion, local_rotation=None):
    frames = motion.frames
    bodies = len(motion.node_names)
    global_q = np.zeros((frames, bodies, 4), dtype=np.float64)
    global_p = np.zeros((frames, bodies, 3), dtype=np.float64)
    local_q = normalize_quat(
        motion.rotation if local_rotation is None else local_rotation
    )
    for body_id in range(bodies):
        parent = int(motion.parent_indices[body_id])
        if parent == -1:
            global_q[:, body_id] = local_q[:, body_id]
            global_p[:, body_id] = motion.root_translation
        else:
            global_q[:, body_id] = normalize_quat(
                quat_mul(global_q[:, parent], local_q[:, body_id])
            )
            offset = np.broadcast_to(motion.local_translation[body_id], (frames, 3))
            global_p[:, body_id] = global_p[:, parent] + quat_rotate(
                global_q[:, parent], offset
            )
    return global_q, global_p


def extract_hinge_dof(motion):
    missing = [body for _, body, _ in DOF_SPECS if body not in motion.node_index]
    if missing:
        raise ValidationError("Target motion is missing hinge bodies: {}".format(sorted(set(missing))))
    out = np.zeros((motion.frames, len(DOF_SPECS)), dtype=np.float64)
    residual = np.zeros_like(out)
    for i, (_, body, axis) in enumerate(DOF_SPECS):
        q = motion.rotation[:, motion.node_index[body]]
        theta = single_axis_angle(q, axis)
        out[:, i] = theta
        residual[:, i] = quat_distance(q, axis_quat(theta, axis))
    return out, residual


def dof_to_obs_numpy(dof_pos):
    if dof_pos.shape[-1] != 28:
        raise ValidationError("Expected 28-D dof_pos, got {}".format(dof_pos.shape))
    chunks = []
    for start, end in zip(DOF_OFFSETS[:-1], DOF_OFFSETS[1:]):
        joint = dof_pos[:, start:end]
        if end - start == 3:
            qx = axis_quat(joint[:, 0], 0)
            qy = axis_quat(joint[:, 1], 1)
            qz = axis_quat(joint[:, 2], 2)
            chunks.append(obs6(normalize_quat(quat_mul(quat_mul(qx, qy), qz))))
        else:
            chunks.append(joint)
    result = np.concatenate(chunks, axis=-1)
    if result.shape[-1] != 52:
        raise ValidationError("Expected 52-D dof obs, got {}".format(result.shape))
    return result


def center_quaternion_sequence(q):
    """Independent NumPy implementation of mean local-orientation removal."""
    q = normalize_quat(q)
    q = q * np.where(q[..., 3:4] < 0.0, -1.0, 1.0)
    mean_q = normalize_quat(np.mean(q, axis=0, keepdims=True))[0]
    mean_inv = quat_conjugate(mean_q)
    centered = normalize_quat(
        quat_mul(np.broadcast_to(mean_inv, q.shape), q)
    )
    return centered, mean_q


def describe_motion(label, motion, lines):
    quat_norm_error = np.max(np.abs(np.linalg.norm(motion.rotation, axis=-1) - 1.0))
    finite = bool(
        np.all(np.isfinite(motion.rotation))
        and np.all(np.isfinite(motion.root_translation))
        and np.all(np.isfinite(motion.local_translation))
    )
    lines.extend([
        "{}:".format(label),
        "  path: {}".format(motion.path),
        "  frames: {}".format(motion.frames),
        "  fps: {:.8g}".format(motion.fps),
        "  duration: {:.8g} s".format(motion.duration),
        "  root_translation shape: {}".format(motion.root_translation.shape),
        "  local_rotation shape: {}".format(motion.rotation.shape),
        "  nodes ({}): {}".format(len(motion.node_names), motion.node_names),
        "  parent_indices: {}".format(motion.parent_indices.tolist()),
        "  finite: {}".format(finite),
        "  max quaternion norm error: {:.9g}".format(quat_norm_error),
    ])
    return finite, float(quat_norm_error)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "max_frame": int(np.argmax(values)),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_xml_joints(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    compiler = root.find("compiler")
    angle_unit = "degree" if compiler is None else compiler.get("angle", "degree").lower()
    if angle_unit not in ("degree", "radian"):
        raise ValidationError("Unsupported MJCF angle unit: {}".format(angle_unit))
    scale = math.pi / 180.0 if angle_unit == "degree" else 1.0

    joints = OrderedDict()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValidationError("XML has no worldbody")
    for elem in worldbody.iter("joint"):
        name = elem.get("name")
        if not name:
            continue
        axis = np.fromstring(elem.get("axis", "0 0 1"), sep=" ", dtype=np.float64)
        range_text = elem.get("range")
        if range_text is None:
            lower, upper = -np.inf, np.inf
        else:
            joint_range = np.fromstring(range_text, sep=" ", dtype=np.float64)
            if joint_range.size != 2:
                raise ValidationError("Invalid range for joint {}".format(name))
            lower, upper = joint_range * scale
        joints[name] = {
            "name": name,
            "axis": axis,
            "lower": float(lower),
            "upper": float(upper),
            "angle_unit": angle_unit,
        }
    return joints, angle_unit


def validate_motionlib(
    target_path, target, expected_pos, expected_vel, summary, comparison_threshold
):
    """Optional dependency-heavy validation against the actual MotionLibV2."""
    try:
        # Isaac Gym must initialize its PyTorch bindings before torch itself is
        # imported in this process. Importing torch first causes Isaac Gym to
        # reject the extension load even when both packages are installed.
        from isaacgym import gymapi, gymtorch  # noqa: F401
        import torch

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from isaacgymenvs.tasks.amp.utils_amp.motion_lib_v2 import MotionLibV2
        from isaacgymenvs.tasks.SRLEvo.humanoid_amp_s1_smpl_base_v2 import dof_to_obs

        key_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        key_ids = np.asarray([target.node_index[name] for name in key_names], dtype=np.int64)
        lib = MotionLibV2(
            motion_file=str(target_path),
            num_dofs=28,
            key_body_ids=key_ids,
            device="cpu",
            dof_representation="hinge_chain",
        )
        motion = lib._motions[0]
        pos = lib._local_rotation_to_dof(torch.from_numpy(target.rotation.astype(np.float32)))
        pos = pos.detach().cpu().numpy()
        vel = np.asarray(motion.dof_vels)
        actual_obs = dof_to_obs(torch.from_numpy(pos.astype(np.float32))).detach().cpu().numpy()
        expected_obs = dof_to_obs_numpy(pos)
        shape_ok = (
            pos.shape == expected_pos.shape
            and vel.shape == expected_vel.shape
            and actual_obs.shape == (target.frames, 52)
        )
        pos_delta = (
            float(np.max(np.abs(wrap_to_pi(pos - expected_pos))))
            if pos.shape == expected_pos.shape
            else float("inf")
        )
        vel_delta = (
            float(np.max(np.abs(vel - expected_vel)))
            if vel.shape == expected_vel.shape
            else float("inf")
        )
        obs_delta = (
            float(np.max(np.abs(actual_obs - expected_obs)))
            if actual_obs.shape == expected_obs.shape
            else float("inf")
        )
        values_ok = max(pos_delta, vel_delta, obs_delta) < comparison_threshold
        passed = shape_ok and values_ok
        summary.extend([
            "MotionLibV2 optional check: {}".format("PASS" if passed else "FAIL"),
            "  Isaac Gym import order: isaacgym -> torch",
            "  dof_pos shape: {}".format(pos.shape),
            "  dof_vel shape: {}".format(vel.shape),
            "  dof_to_obs shape: {}".format(actual_obs.shape),
            "  comparison threshold: {:.9g}".format(comparison_threshold),
            "  max dof_pos delta vs pure NumPy: {:.9g}".format(pos_delta),
            "  max dof_vel delta vs wrapped finite difference: {:.9g}".format(vel_delta),
            "  max dof_to_obs delta vs pure NumPy: {:.9g}".format(obs_delta),
        ])
        return passed
    except Exception as exc:
        summary.append("MotionLibV2 optional check: SKIPPED ({})".format(repr(exc)))
        return None


def make_plots(output_dir, rotation_series, dof_pos, dof_vel, old_pos, new_pos, body_pos_error):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return "Plots skipped: {}".format(repr(exc))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for ax, (name, values) in zip(axes.flat, rotation_series.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("frame")
    axes[-1, 1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "conceptual_joint_rotation_error.png"), dpi=150)
    plt.close(fig)

    for data, filename, ylabel in [
        (dof_pos, "hinge_angles.png", "angle [rad]"),
        (dof_vel, "hinge_velocities.png", "velocity [rad/s]"),
    ]:
        fig, axes = plt.subplots(7, 4, figsize=(15, 18), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(data[:, i], linewidth=0.8)
            ax.set_title(DOF_SPECS[i][0], fontsize=8)
            ax.grid(True, alpha=0.2)
        fig.supylabel(ylabel)
        fig.supxlabel("frame")
        fig.tight_layout()
        fig.savefig(str(plots_dir / filename), dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, foot in enumerate(["right_foot", "left_foot"]):
        old_rel = old_pos[foot]
        new_rel = new_pos[foot]
        for axis in range(3):
            axes[row, axis].plot(old_rel[:, axis], label="old")
            axes[row, axis].plot(new_rel[:, axis], "--", label="v2")
            axes[row, axis].set_title("{} {}".format(foot, "xyz"[axis]))
            axes[row, axis].grid(True, alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(str(plots_dir / "feet_relative_to_pelvis.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, values in body_pos_error.items():
        ax.plot(values, label=name, linewidth=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "real_body_position_errors.png"), dpi=150)
    plt.close(fig)
    return "Plots written to {}".format(plots_dir)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-motion",
        default=str(repo_root / "assets/amp/motions/amp_humanoid_walk_175.npy"),
    )
    parser.add_argument(
        "--target-motion",
        default=str(repo_root / "assets/amp/motions/amp_humanoid_walk_175_v2.npy"),
    )
    parser.add_argument(
        "--target-tpose",
        default=str(script_dir / "data/amp_humanoid_175_v2_tpose.npy"),
    )
    parser.add_argument(
        "--xml",
        default=str(repo_root / "assets/mjcf/amp_humanoid_175_v2.xml"),
    )
    parser.add_argument("--output-dir", default=str(script_dir / "validation_results"))
    parser.add_argument("--rotation-threshold", type=float, default=1e-4)
    parser.add_argument("--obs-threshold", type=float, default=1e-4)
    parser.add_argument("--position-rmse-threshold", type=float, default=1e-4)
    parser.add_argument("--quat-norm-threshold", type=float, default=1e-5)
    parser.add_argument("--per-frame-jump-warning", type=float, default=0.5)
    parser.add_argument("--velocity-warning", type=float, default=20.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-motionlib", action="store_true")
    parser.add_argument(
        "--center-neck-reference",
        action="store_true",
        help=(
            "Expect the target motion to remove the source head's mean local "
            "orientation before splitting it into neck X/Y/Z hinges."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source_motion).resolve()
    target_path = Path(args.target_motion).resolve()
    tpose_path = Path(args.target_tpose).resolve()
    xml_path = Path(args.xml).resolve()
    for label, path in [
        ("source motion", source_path),
        ("target motion", target_path),
        ("target T-pose", tpose_path),
        ("target XML", xml_path),
    ]:
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    old = MotionData(source_path)
    new = MotionData(target_path)
    tpose = TposeData(tpose_path)
    summary = ["V2 MOTION CONVERSION VALIDATION", "=" * 80]
    failures = []
    warnings = []

    old_finite, old_norm_error = describe_motion("source", old, summary)
    new_finite, new_norm_error = describe_motion("target", new, summary)
    if old.frames != new.frames:
        failures.append("Source/target frame counts differ")
    if not np.isclose(old.fps, new.fps, atol=1e-8):
        failures.append("Source/target FPS differ")
    if not old_finite or not new_finite:
        failures.append("NaN or Inf found in motion data")
    if max(old_norm_error, new_norm_error) >= args.quat_norm_threshold:
        failures.append("Quaternion norm error exceeds threshold")

    missing_old = [old_name for old_name, _ in JOINT_GROUPS.values() if old_name not in old.node_index]
    missing_new = [
        node for _, new_nodes in JOINT_GROUPS.values() for node in new_nodes
        if node not in new.node_index
    ]
    missing_real = [
        body for body in REAL_BODIES
        if body not in old.node_index or body not in new.node_index
    ]
    if missing_old or missing_new or missing_real:
        raise ValidationError(
            "Missing skeleton nodes: old={}, new={}, real={}".format(
                missing_old, missing_new, missing_real
            )
        )

    rotation_rows = []
    obs_rows = []
    rotation_series = OrderedDict()
    old_obs_by_joint = {}
    new_obs_by_joint = {}
    expected_old_rotation = old.rotation.copy()
    if args.center_neck_reference:
        head_id = old.node_index["head"]
        centered_head, neck_mean_q = center_quaternion_sequence(
            expected_old_rotation[:, head_id]
        )
        expected_old_rotation[:, head_id] = centered_head
        neck_before = np.rad2deg(
            quat_to_hinge_xyz_chain(old.rotation[:, head_id])
        )
        neck_after = np.rad2deg(quat_to_hinge_xyz_chain(centered_head))
        summary.extend([
            "",
            "neck rest-pose compensation expected:",
            "  removed mean quaternion: {}".format(neck_mean_q),
            "  component-angle mean before [deg]: {}".format(
                np.mean(neck_before, axis=0)
            ),
            "  component-angle mean after [deg]: {}".format(
                np.mean(neck_after, axis=0)
            ),
        ])
    max_matrix_cross_error = 0.0
    for name, (old_body, new_nodes) in JOINT_GROUPS.items():
        q_old = expected_old_rotation[:, old.node_index[old_body]]
        qx = new.rotation[:, new.node_index[new_nodes[0]]]
        qy = new.rotation[:, new.node_index[new_nodes[1]]]
        qz = new.rotation[:, new.node_index[new_nodes[2]]]
        q_recon = normalize_quat(quat_mul(quat_mul(qx, qy), qz))
        error = quat_distance(q_old, q_recon)
        rotation_series[name] = error
        values = stats(error)
        rotation_rows.append(dict(joint=name, **values))

        # Independent matrix-chain cross-check catches quaternion product mistakes.
        ax = single_axis_angle(qx, 0)
        ay = single_axis_angle(qy, 1)
        az = single_axis_angle(qz, 2)
        matrix_chain = np.matmul(np.matmul(axis_matrix(ax, 0), axis_matrix(ay, 1)), axis_matrix(az, 2))
        matrix_delta = np.max(np.abs(matrix_chain - quat_to_matrix(q_recon)))
        max_matrix_cross_error = max(max_matrix_cross_error, float(matrix_delta))

        old_feature = obs6(q_old)
        new_feature = obs6(q_recon)
        old_obs_by_joint[name] = old_feature
        new_obs_by_joint[name] = new_feature
        obs_error = np.linalg.norm(old_feature - new_feature, axis=-1)
        obs_rows.append({
            "joint": name,
            "mean_l2": float(np.mean(obs_error)),
            "rms_l2": float(np.sqrt(np.mean(obs_error ** 2))),
            "max_l2": float(np.max(obs_error)),
            "max_frame": int(np.argmax(obs_error)),
        })

    max_rotation_error = max(row["max"] for row in rotation_rows)
    max_obs_error = max(row["max_l2"] for row in obs_rows)
    if max_rotation_error >= args.rotation_threshold:
        failures.append("Conceptual 3DoF reconstruction error exceeds threshold")
    if max_obs_error >= args.obs_threshold:
        failures.append("Conceptual 3DoF 6D observation error exceeds threshold")
    if max_matrix_cross_error >= 1e-10:
        failures.append("Quaternion and rotation-matrix chain composition disagree")

    scalar_rows = []
    for name, (body, axis) in SCALAR_JOINTS.items():
        q_old = old.rotation[:, old.node_index[body]]
        theta = single_axis_angle(q_old, axis)
        residual = quat_distance(q_old, axis_quat(theta, axis))
        values = stats(residual)
        scalar_rows.append(dict(joint=name, **values))
        if values["max"] >= args.rotation_threshold:
            warnings.append("{} source rotation is not pure Y-axis".format(name))

    old_global_q, old_global_p = forward_kinematics(
        old, local_rotation=expected_old_rotation
    )
    new_global_q, new_global_p = forward_kinematics(new)
    old_pelvis = old_global_p[:, old.node_index["pelvis"]]
    new_pelvis = new_global_p[:, new.node_index["pelvis"]]
    global_rows = []
    old_relative = {}
    new_relative = {}
    body_pos_error = {}
    for body in REAL_BODIES:
        old_id = old.node_index[body]
        new_id = new.node_index[body]
        old_rel = old_global_p[:, old_id] - old_pelvis
        new_rel = new_global_p[:, new_id] - new_pelvis
        old_relative[body] = old_rel
        new_relative[body] = new_rel
        pos_error = np.linalg.norm(old_rel - new_rel, axis=-1)
        rot_error = quat_distance(old_global_q[:, old_id], new_global_q[:, new_id])
        body_pos_error[body] = pos_error
        global_rows.append({
            "body": body,
            "position_rmse": float(np.sqrt(np.mean(pos_error ** 2))),
            "position_max": float(np.max(pos_error)),
            "position_max_frame": int(np.argmax(pos_error)),
            "rotation_mean": float(np.mean(rot_error)),
            "rotation_max": float(np.max(rot_error)),
            "rotation_max_frame": int(np.argmax(rot_error)),
        })
    max_position_rmse = max(row["position_rmse"] for row in global_rows)
    max_global_rotation = max(row["rotation_max"] for row in global_rows)
    if max_position_rmse >= args.position_rmse_threshold:
        failures.append("Real-body relative-position RMSE exceeds threshold")
    if max_global_rotation >= args.rotation_threshold:
        failures.append("Real-body global-rotation error exceeds threshold")

    root_delta = new.root_translation - old.root_translation
    root_xy_std = np.std(root_delta[:, :2], axis=0)
    root_z_std = float(np.std(root_delta[:, 2]))
    summary.extend([
        "",
        "root translation delta:",
        "  mean: {}".format(np.mean(root_delta, axis=0)),
        "  std: {}".format(np.std(root_delta, axis=0)),
    ])
    if np.max(root_xy_std) > args.position_rmse_threshold or root_z_std > args.position_rmse_threshold:
        warnings.append("Root translation difference is not approximately constant")

    tpose_rows = []
    for node in INSERTED_NODES:
        if node not in tpose.node_index:
            failures.append("Target T-pose missing inserted node {}".format(node))
            continue
        q = tpose.rotation[tpose.node_index[node]]
        angle_error = float(quat_distance(q[None, :], np.asarray([[0.0, 0.0, 0.0, 1.0]]))[0])
        tpose_rows.append({"node": node, "identity_angle_error": angle_error})
        if angle_error >= args.rotation_threshold:
            failures.append("Inserted node {} has non-identity T-pose rotation".format(node))
    summary.extend(["", "inserted-node T-pose identity errors:"])
    summary.extend([
        "  {}: {:.9g} rad".format(row["node"], row["identity_angle_error"])
        for row in tpose_rows
    ])

    dof_pos, hinge_residual = extract_hinge_dof(new)
    if np.max(hinge_residual) >= args.rotation_threshold:
        failures.append("Target local hinge quaternion contains off-axis rotation")
    dof_obs = dof_to_obs_numpy(dof_pos)
    old_full_obs = []
    new_full_obs = []
    for start, end in zip(DOF_OFFSETS[:-1], DOF_OFFSETS[1:]):
        if end - start == 3:
            # Derive the conceptual joint name from the actual 28-D DOF order.
            # This must not depend on JOINT_GROUPS insertion order.
            dof_name = DOF_SPECS[start][0]
            if not dof_name.endswith("_x"):
                raise ValidationError(
                    "Expected a 3DoF group to start with an _x joint, got {}".format(dof_name)
                )
            joint_name = dof_name[:-2]
            old_full_obs.append(old_obs_by_joint[joint_name])
            new_full_obs.append(new_obs_by_joint[joint_name])
        else:
            name, body, axis = DOF_SPECS[start]
            old_angle = single_axis_angle(
                expected_old_rotation[:, old.node_index[body]], axis
            )
            old_full_obs.append(old_angle[:, None])
            new_full_obs.append(dof_pos[:, start:end])
    old_full_obs = np.concatenate(old_full_obs, axis=-1)
    new_full_obs = np.concatenate(new_full_obs, axis=-1)
    reconstructed_obs_layout_error = np.max(np.abs(new_full_obs - dof_obs))
    expert_obs_error = np.linalg.norm(old_full_obs - dof_obs, axis=-1)
    if dof_obs.shape[-1] != 52 or old_full_obs.shape[-1] != 52:
        failures.append("End-to-end joint observation is not 52-D")
    if reconstructed_obs_layout_error >= args.obs_threshold:
        failures.append("Reconstructed v2 observation order differs from dof_to_obs order")
    if np.max(expert_obs_error) >= args.obs_threshold:
        failures.append("Full expert 52-D observation differs from legacy motion")

    xml_joints, angle_unit = parse_xml_joints(xml_path)
    xml_order = [name for name in xml_joints if name in {spec[0] for spec in DOF_SPECS}]
    expected_order = [spec[0] for spec in DOF_SPECS]
    if xml_order != expected_order:
        failures.append("XML hinge order differs from expected 28-D DOF order")
    limit_rows = []
    tolerance = 1e-8
    for i, (name, _, _) in enumerate(DOF_SPECS):
        if name not in xml_joints:
            raise ValidationError("XML is missing joint {}".format(name))
        info = xml_joints[name]
        values = dof_pos[:, i]
        low_mask = values < info["lower"] - tolerance
        high_mask = values > info["upper"] + tolerance
        max_violation = max(
            float(np.max(info["lower"] - values[low_mask])) if np.any(low_mask) else 0.0,
            float(np.max(values[high_mask] - info["upper"])) if np.any(high_mask) else 0.0,
        )
        limit_rows.append({
            "joint": name,
            "axis_x": info["axis"][0],
            "axis_y": info["axis"][1],
            "axis_z": info["axis"][2],
            "xml_angle_unit": angle_unit,
            "motion_min": float(np.min(values)),
            "motion_max": float(np.max(values)),
            "xml_lower_rad": info["lower"],
            "xml_upper_rad": info["upper"],
            "lower_violation_frames": int(np.sum(low_mask)),
            "lower_violation_ratio": float(np.mean(low_mask)),
            "upper_violation_frames": int(np.sum(high_mask)),
            "upper_violation_ratio": float(np.mean(high_mask)),
            "max_violation_rad": max_violation,
        })
    total_limit_violations = sum(
        row["lower_violation_frames"] + row["upper_violation_frames"] for row in limit_rows
    )
    if total_limit_violations:
        warnings.append("Motion violates XML joint limits in {} frame-joint samples".format(total_limit_violations))

    dt = 1.0 / new.fps
    per_frame_delta = wrap_to_pi(np.diff(dof_pos, axis=0))
    dof_vel = per_frame_delta / dt
    # MotionLib repeats the final finite-difference velocity for the last frame.
    dof_vel_full = np.concatenate([dof_vel, dof_vel[-1:]], axis=0)
    velocity_rows = []
    for i, (name, _, _) in enumerate(DOF_SPECS):
        abs_vel = np.abs(dof_vel[:, i])
        abs_jump = np.abs(per_frame_delta[:, i])
        velocity_rows.append({
            "joint": name,
            "max_frame_delta": float(np.max(abs_jump)),
            "delta_max_frame": int(np.argmax(abs_jump)),
            "velocity_mean": float(np.mean(dof_vel[:, i])),
            "velocity_std": float(np.std(dof_vel[:, i])),
            "abs_velocity_p95": float(np.percentile(abs_vel, 95)),
            "abs_velocity_p99": float(np.percentile(abs_vel, 99)),
            "max_abs_velocity": float(np.max(abs_vel)),
            "velocity_max_frame": int(np.argmax(abs_vel)),
        })
        if np.max(abs_jump) > args.per_frame_jump_warning:
            warnings.append("{} has a per-frame angle jump above threshold".format(name))
        if np.max(abs_vel) > args.velocity_warning:
            warnings.append("{} has velocity above threshold".format(name))

    if not args.skip_motionlib:
        motionlib_result = validate_motionlib(
            target_path,
            new,
            dof_pos,
            dof_vel_full,
            summary,
            max(args.obs_threshold, 1e-5),
        )
        if motionlib_result is False:
            failures.append("MotionLibV2 end-to-end comparison failed")
    else:
        summary.append("MotionLibV2 optional check: SKIPPED by command line")

    rotation_fields = ["joint", "mean", "rms", "median", "p95", "p99", "max", "max_frame"]
    write_csv(output_dir / "joint_rotation_errors.csv", rotation_rows + scalar_rows, rotation_fields)
    write_csv(
        output_dir / "observation_6d_errors.csv",
        obs_rows,
        ["joint", "mean_l2", "rms_l2", "max_l2", "max_frame"],
    )
    write_csv(
        output_dir / "global_body_errors.csv",
        global_rows,
        [
            "body", "position_rmse", "position_max", "position_max_frame",
            "rotation_mean", "rotation_max", "rotation_max_frame",
        ],
    )
    write_csv(
        output_dir / "joint_limit_report.csv",
        limit_rows,
        [
            "joint", "axis_x", "axis_y", "axis_z", "xml_angle_unit",
            "motion_min", "motion_max", "xml_lower_rad", "xml_upper_rad",
            "lower_violation_frames", "lower_violation_ratio",
            "upper_violation_frames", "upper_violation_ratio", "max_violation_rad",
        ],
    )
    write_csv(
        output_dir / "velocity_report.csv",
        velocity_rows,
        [
            "joint", "max_frame_delta", "delta_max_frame", "velocity_mean",
            "velocity_std", "abs_velocity_p95", "abs_velocity_p99",
            "max_abs_velocity", "velocity_max_frame",
        ],
    )

    if not args.no_plots:
        summary.append(
            make_plots(
                output_dir,
                rotation_series,
                dof_pos,
                dof_vel_full,
                old_relative,
                new_relative,
                body_pos_error,
            )
        )
    else:
        summary.append("Plots skipped by command line")

    summary.extend([
        "",
        "CORE METRICS",
        "  max conceptual rotation error: {:.9g} rad".format(max_rotation_error),
        "  max conceptual 6D error: {:.9g}".format(max_obs_error),
        "  max full 52-D expert obs frame error: {:.9g}".format(np.max(expert_obs_error)),
        "  max reconstructed-vs-dof_to_obs layout delta: {:.9g}".format(
            reconstructed_obs_layout_error
        ),
        "  max quaternion/matrix composition delta: {:.9g}".format(max_matrix_cross_error),
        "  max real-body position RMSE: {:.9g} m".format(max_position_rmse),
        "  max real-body global rotation error: {:.9g} rad".format(max_global_rotation),
        "  XML angle unit: {}".format(angle_unit),
        "  joint-limit violation samples: {}".format(total_limit_violations),
        "  max hinge residual rotation: {:.9g} rad".format(np.max(hinge_residual)),
    ])

    if failures:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARNING"
    else:
        verdict = "PASS"
    summary.extend(["", "VERDICT: {}".format(verdict)])
    if failures:
        summary.append("Failures:")
        summary.extend(["  - {}".format(item) for item in sorted(set(failures))])
    if warnings:
        summary.append("Warnings:")
        summary.extend(["  - {}".format(item) for item in sorted(set(warnings))])

    categories = []
    combined_messages = " ".join(failures + warnings).lower()
    if "reconstruction" in combined_messages or "composition" in combined_messages:
        categories.append("quaternion multiplication order / XYZ decomposition branch")
    if "t-pose" in combined_messages or "off-axis" in combined_messages:
        categories.append("T-pose/rest rotation")
    if "real-body" in combined_messages or "missing" in combined_messages:
        categories.append("skeleton node mapping")
    if "joint limit" in combined_messages:
        categories.append("joint limits")
    if "velocity" in combined_messages or "jump" in combined_messages:
        categories.append("velocity discontinuity")
    if "root translation" in combined_messages:
        categories.append("grounding/root translation")
    if "dof order" in combined_messages or "motionlib" in combined_messages:
        categories.append("MotionLib DOF order")
    if categories:
        summary.append("Likely issue categories:")
        summary.extend(["  - {}".format(item) for item in categories])

    summary_path = output_dir / "summary.txt"
    summary_text = "\n".join(summary) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("Reports written to: {}".format(output_dir))
    return 2 if failures else (1 if warnings else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("VALIDATION ERROR: {}".format(exc), file=sys.stderr)
        raise
