from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

from mujoco_ppo.srl_mujoco_v3_env import SRLMujocoV3Env, V3WalkEnvConfig


CONNECTOR_JOINT_NAMES: Tuple[str, str] = (
    "left_hip_connector_y_slide",
    "right_hip_connector_y_slide",
)


@dataclass
class V4WalkEnvConfig(V3WalkEnvConfig):
    """V3 policy interface with compliant hip connectors and no battery."""

    xml_path: str = "mjcf/srl_real_v1/srl_real_bot_nobattery.xml"
    connector_stiffness: float = 94_000.0
    connector_damping: float = 200.0
    dr_connector_stiffness_range: Tuple[float, float] = (1.0, 1.0)
    dr_connector_damping_range: Tuple[float, float] = (1.0, 1.0)


class SRLMujocoV4Env(SRLMujocoV3Env):
    """133D/153D asymmetric environment for the compliant v4 mechanism.

    The policy still observes and actuates exactly the original six rotary
    joints. The two connector slides are passive internal states integrated by
    MuJoCo and intentionally excluded from actor/critic observations.
    """

    def __init__(self, config: Optional[V4WalkEnvConfig] = None):
        resolved_config = config or V4WalkEnvConfig()
        for name, value_range in (
            (
                "dr_connector_stiffness_range",
                resolved_config.dr_connector_stiffness_range,
            ),
            (
                "dr_connector_damping_range",
                resolved_config.dr_connector_damping_range,
            ),
        ):
            lo, hi = map(float, value_range)
            if lo <= 0.0 or hi < lo:
                raise ValueError(f"{name} must be positive and ordered.")
        super().__init__(resolved_config)
        self.connector_joint_ids = np.asarray(
            [self._require_joint(name) for name in CONNECTOR_JOINT_NAMES],
            dtype=np.int32,
        )
        self.connector_qpos_indices = self.model.jnt_qposadr[
            self.connector_joint_ids
        ].astype(np.int32, copy=True)
        self.connector_dof_indices = self.model.jnt_dofadr[
            self.connector_joint_ids
        ].astype(np.int32, copy=True)
        self.model.jnt_stiffness[self.connector_joint_ids] = float(
            self.cfg.connector_stiffness
        )
        self.model.dof_damping[self.connector_dof_indices] = float(
            self.cfg.connector_damping
        )
        self.nominal_connector_stiffness = self.model.jnt_stiffness[
            self.connector_joint_ids
        ].copy()
        self.nominal_connector_damping = self.model.dof_damping[
            self.connector_dof_indices
        ].copy()
        # DR restores values from this snapshot at every reset, so capture the
        # explicitly configured connector properties as the v4 nominal model.
        self._store_nominal_model_params()

    def _reset_randomized_params(self):
        super()._reset_randomized_params()
        stiffness_scale = 1.0
        damping_scale = 1.0
        if self.cfg.domain_randomization_enable:
            stiffness_scale = self._uniform(
                self.cfg.dr_connector_stiffness_range
            )
            damping_scale = self._uniform(self.cfg.dr_connector_damping_range)
        self.model.jnt_stiffness[self.connector_joint_ids] = (
            self.nominal_connector_stiffness * stiffness_scale
        )
        self.model.dof_damping[self.connector_dof_indices] = (
            self.nominal_connector_damping * damping_scale
        )
        self.dr_sampled_params.update(
            {
                "connector_stiffness_scale": float(stiffness_scale),
                "connector_damping_scale": float(damping_scale),
                "connector_stiffness_n_per_m": float(
                    self.model.jnt_stiffness[self.connector_joint_ids[0]]
                ),
                "connector_damping_ns_per_m": float(
                    self.model.dof_damping[self.connector_dof_indices[0]]
                ),
            }
        )

    def set_gait_period(self, gait_period: int):
        """Change gait period while preserving the current phase angle."""
        new_period = max(int(gait_period), 2)
        old_period = max(int(self.cfg.gait_period), 2)
        if new_period == old_period:
            return
        phase_fraction = float(self.phase_counter) / float(old_period)
        self.cfg.gait_period = float(new_period)
        self.phase_counter = int(round(phase_fraction * new_period)) % new_period

    def _require_joint(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(
                f"V4 XML is missing required passive connector joint: {name}"
            )
        return int(joint_id)

    def _get_info(self):
        info = super()._get_info()
        connector_pos = self.data.qpos[self.connector_qpos_indices].copy()
        connector_vel = self.data.qvel[self.connector_dof_indices].copy()
        connector_stiffness = self.model.jnt_stiffness[
            self.connector_joint_ids
        ]
        connector_damping = self.model.dof_damping[self.connector_dof_indices]
        connector_force = -(
            connector_stiffness * connector_pos
            + connector_damping * connector_vel
        )
        info.update(
            {
                "connector_slide_pos": connector_pos,
                "connector_slide_vel": connector_vel,
                "connector_slide_force": connector_force,
                "connector_slide_abs_max": float(np.max(np.abs(connector_pos))),
                "connector_stiffness": connector_stiffness.copy(),
                "connector_damping": connector_damping.copy(),
            }
        )
        return info
