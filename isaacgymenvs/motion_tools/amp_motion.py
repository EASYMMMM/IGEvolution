"""Read/write the repository's Poselib SkeletonMotion format without Torch."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
MOTIONS = PROJECT / "assets/amp/motions"
MJCF = PROJECT / "assets/mjcf"
DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]


def load_motion(path):
    # Repository AMP files are trusted local NumPy pickle dictionaries.
    data = np.load(path, allow_pickle=True).item()
    if data.get("__name__") != "SkeletonMotion" or not data["is_local"]:
        raise ValueError("Expected a local-rotation Poselib SkeletonMotion")
    return data


def tensor(arr, dtype="float32"):
    return {"arr": np.asarray(arr, dtype=dtype), "context": {"dtype": dtype}}


def continuous_quaternions(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for frame in range(1, len(q)):
        flip = np.sum(q[frame] * q[frame - 1], axis=-1) < 0
        q[frame, flip] *= -1
    return q


def forward_kinematics(rotation, root_translation, tree):
    local = Rotation.from_quat(np.asarray(rotation).reshape(-1, 4)).as_matrix()
    local = local.reshape(*rotation.shape[:2], 3, 3)
    parents = tree["parent_indices"]["arr"]
    offsets = tree["local_translation"]["arr"]
    global_rot = np.empty_like(local)
    pos = np.empty((*rotation.shape[:2], 3), dtype=np.float64)
    for j, parent in enumerate(parents):
        if parent < 0:
            global_rot[:, j] = local[:, j]
            pos[:, j] = root_translation
        else:
            global_rot[:, j] = global_rot[:, parent] @ local[:, j]
            pos[:, j] = pos[:, parent] + np.einsum("fij,j->fi", global_rot[:, parent], offsets[j])
    return pos, global_rot


def make_motion(rotation, root_translation, tree, fps):
    rotation = continuous_quaternions(rotation)
    pos, matrices = forward_kinematics(rotation, root_translation, tree)
    velocity = gaussian_filter1d(np.gradient(pos, axis=0), 2, axis=0, mode="nearest") * fps
    global_q = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(*rotation.shape)
    angular = np.zeros_like(pos)
    for j in range(rotation.shape[1]):
        # Poselib stores angular velocity in world coordinates.
        r = Rotation.from_quat(global_q[:, j])
        angular[:-1, j] = (r[1:] * r[:-1].inv()).as_rotvec() * fps
    angular = gaussian_filter1d(angular, 2, axis=0, mode="nearest")
    return {"rotation": tensor(rotation), "root_translation": tensor(root_translation),
            "global_velocity": tensor(velocity), "global_angular_velocity": tensor(angular),
            "skeleton_tree": tree, "is_local": True, "fps": np.array(fps, dtype=np.int32),
            "__name__": "SkeletonMotion"}


def dof_positions(rotation):
    result = np.zeros((len(rotation), 28), dtype=np.float32)
    for i, body in enumerate(DOF_BODY_IDS):
        a, b = DOF_OFFSETS[i:i + 2]
        v = Rotation.from_quat(rotation[:, body]).as_rotvec()
        result[:, a:b] = v if b - a == 3 else v[:, 1:2]
    return result
