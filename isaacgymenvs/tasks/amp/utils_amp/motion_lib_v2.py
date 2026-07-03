"""v2 MotionLib workspace for chain-XML migration.

``legacy_expmap`` keeps the original IsaacGym AMP semantics.

``hinge_chain`` converts SkeletonMotion local joint rotations into raw hinge
coordinates that match the chain-XML route:

- If a conceptual 3DoF joint has already been split into explicit x/y/z link
  bodies, we read each hinge angle from those link-local quaternions.
- If a compatible skeleton still stores a conceptual joint as one body-local
  quaternion, we decompose it using the same Qx * Qy * Qz convention as the
  explicit nested hinge chain.
"""

import numpy as np
import torch

from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import normalize_angle


class MotionLibV2(MotionLib):
    def __init__(self, motion_file, num_dofs, key_body_ids, device,
                 dof_representation="legacy_expmap"):
        self._dof_representation = dof_representation
        self._hinge_chain_specs = None
        self._hinge_chain_name_signature = None
        super().__init__(motion_file=motion_file,
                         num_dofs=num_dofs,
                         key_body_ids=key_body_ids,
                         device=device)

    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)
        if self._dof_representation == "hinge_chain" and len(self._motions) > 0:
            self._ensure_hinge_chain_specs(self._motions[0].skeleton_tree)

    def _compute_motion_dof_vels(self, motion):
        if self._dof_representation == "hinge_chain":
            self._ensure_hinge_chain_specs(motion.skeleton_tree)
        return super()._compute_motion_dof_vels(motion)

    def _local_rotation_to_dof(self, local_rot):
        if self._dof_representation == "legacy_expmap":
            return super()._local_rotation_to_dof(local_rot)
        if self._dof_representation == "hinge_chain":
            return self._hinge_chain_local_rotation_to_dof(local_rot)

        raise ValueError(f"Unsupported dof_representation={self._dof_representation}")

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        if self._dof_representation == "legacy_expmap":
            return super()._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        if self._dof_representation == "hinge_chain":
            return self._hinge_chain_local_rotation_to_dof_vel(local_rot0, local_rot1, dt)

        raise ValueError(f"Unsupported dof_representation={self._dof_representation}")

    def _ensure_hinge_chain_specs(self, skeleton_tree):
        node_indices = skeleton_tree._node_indices
        name_signature = tuple(node_indices.keys())
        if self._hinge_chain_specs is not None and self._hinge_chain_name_signature == name_signature:
            return

        self._hinge_chain_specs = self._build_hinge_chain_specs(node_indices)
        self._hinge_chain_name_signature = name_signature

    def _build_hinge_chain_specs(self, node_indices):
        names = set(node_indices.keys())
        specs = []

        def add_group_xyz(start_idx, grouped_body, chain_bodies=None):
            chain_bodies = chain_bodies or []
            if len(chain_bodies) == 3 and all(name in names for name in chain_bodies):
                for axis_idx, body_name in enumerate(chain_bodies):
                    specs.append({
                        "type": "scalar",
                        "body_id": int(node_indices[body_name]),
                        "axis_idx": axis_idx,
                        "out_idx": start_idx + axis_idx,
                        "label": f"{body_name}[axis={axis_idx}]",
                    })
                return

            if grouped_body not in names:
                raise KeyError(
                    f"Cannot resolve grouped 3DoF body '{grouped_body}' and chain bodies "
                    f"{chain_bodies} are not all present either."
                )

            specs.append({
                "type": "group_xyz",
                "body_id": int(node_indices[grouped_body]),
                "out_slice": (start_idx, start_idx + 3),
                "label": grouped_body,
            })

        def add_scalar(out_idx, body_name, axis_idx):
            if body_name not in names:
                raise KeyError(f"Cannot resolve hinge-chain body '{body_name}'")
            specs.append({
                "type": "scalar",
                "body_id": int(node_indices[body_name]),
                "axis_idx": axis_idx,
                "out_idx": out_idx,
                "label": f"{body_name}[axis={axis_idx}]",
            })

        # abdomen / neck / shoulders: support both grouped-body and explicit chain forms
        add_group_xyz(0, "torso", ["abdomen_x_link", "abdomen_y_link", "torso"])
        add_group_xyz(3, "head", ["neck_x_link", "neck_y_link", "head"])
        add_group_xyz(6, "right_upper_arm", ["right_shoulder_x_link", "right_shoulder_y_link", "right_upper_arm"])
        add_scalar(9, "right_lower_arm", 1)
        add_group_xyz(10, "left_upper_arm", ["left_shoulder_x_link", "left_shoulder_y_link", "left_upper_arm"])
        add_scalar(13, "left_lower_arm", 1)

        # hips / ankles in the current v2 XML are explicit hinge chains
        add_scalar(14, "right_hip_x_link", 0)
        add_scalar(15, "right_hip_y_link", 1)
        add_scalar(16, "right_thigh", 2)
        add_scalar(17, "right_shin", 1)
        add_scalar(18, "right_ankle_x_link", 0)
        add_scalar(19, "right_ankle_y_link", 1)
        add_scalar(20, "right_foot", 2)

        add_scalar(21, "left_hip_x_link", 0)
        add_scalar(22, "left_hip_y_link", 1)
        add_scalar(23, "left_thigh", 2)
        add_scalar(24, "left_shin", 1)
        add_scalar(25, "left_ankle_x_link", 0)
        add_scalar(26, "left_ankle_y_link", 1)
        add_scalar(27, "left_foot", 2)

        return specs

    def _hinge_chain_local_rotation_to_dof(self, local_rot):
        if self._hinge_chain_specs is None:
            raise RuntimeError("hinge-chain specs have not been initialized")

        local_rot, squeeze = self._ensure_batched_local_rot(local_rot)
        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float32, device=local_rot.device)

        for spec in self._hinge_chain_specs:
            body_q = local_rot[:, spec["body_id"]]
            if spec["type"] == "group_xyz":
                s, e = spec["out_slice"]
                dof_pos[:, s:e] = self._quat_to_hinge_xyz_chain(body_q)
            elif spec["type"] == "scalar":
                dof_pos[:, spec["out_idx"]] = self._quat_to_single_axis_angle(body_q, spec["axis_idx"])
            else:
                raise ValueError(f"Unknown hinge-chain spec type: {spec['type']}")

        if squeeze:
            return dof_pos[0]
        return dof_pos

    def _hinge_chain_local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        q0 = self._hinge_chain_local_rotation_to_dof(local_rot0)
        q1 = self._hinge_chain_local_rotation_to_dof(local_rot1)

        if q0.dim() > 1:
            q0 = q0[0]
        if q1.dim() > 1:
            q1 = q1[0]

        dq = normalize_angle(q1 - q0) / dt
        return dq.detach().cpu().numpy()

    @staticmethod
    def _ensure_batched_local_rot(local_rot):
        if local_rot.dim() == 2:
            return local_rot.unsqueeze(0), True
        return local_rot, False

    @staticmethod
    def _quat_to_single_axis_angle(q, axis_idx):
        """Extract a signed pure-axis hinge angle without acos near zero."""
        norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
        qn = q / norm
        qn = torch.where(qn[..., 3:4] < 0.0, -qn, qn)
        joint_theta = 2.0 * torch.atan2(qn[..., axis_idx], qn[..., 3])
        return normalize_angle(joint_theta)

    @staticmethod
    def _quat_to_hinge_xyz_chain(q):
        """Decompose Qx(qx) * Qy(qy) * Qz(qz) into nested hinge angles."""
        norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
        qn = q / norm

        x = qn[..., 0]
        y = qn[..., 1]
        z = qn[..., 2]
        w = qn[..., 3]

        t0 = 2.0 * (w * x - y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        ex = torch.atan2(t0, t1)

        t2 = 2.0 * (x * z + w * y)
        t2 = torch.clamp(t2, -1.0, 1.0)
        ey = torch.asin(t2)

        t3 = 2.0 * (w * z - x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        ez = torch.atan2(t3, t4)

        euler = torch.stack([ex, ey, ez], dim=-1)
        return normalize_angle(euler)
