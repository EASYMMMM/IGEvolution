import numpy as np
import torch
from gym import spaces

from isaacgymenvs.tasks.SRLEvo.srl_real_bot import SRL_Real_Bot
from isaacgymenvs.tasks.SRLEvo.stable_gait_rewards import StableGaitRewardMixin
from isaacgymenvs.utils.torch_jit_utils import quat_rotate_inverse


class SRL_Real_Bot_ConcurrentLatent(StableGaitRewardMixin, SRL_Real_Bot):
    """SRL task exposing supervision slots for the concurrent latent encoder."""

    deployable_obs_dim = 133
    privileged_dim = 4
    auxiliary_slot_dim = 16
    com_cop_dim = 3
    contact_dim = 2

    def __init__(self, cfg, *args, **kwargs):
        cfg["env"]["srl_policy_obs_remove_ids"] = [0, 1, 2, 3]
        cfg["env"]["append_current_privileged_obs"] = True
        self._latent_body_masses = None
        self._latent_contact_force_threshold = float(
            cfg["env"].get("latent_contact_force_threshold", 5.0)
        )
        self._latent_contact_height_threshold = float(
            cfg["env"].get("latent_contact_height_threshold", 0.055)
        )
        self._latent_track_randomized_body_masses = bool(
            cfg["env"].get("latent_track_randomized_body_masses", True)
        )
        super().__init__(cfg, *args, **kwargs)

        if self.num_obs != 137 or self.srl_full_obs_size != 153:
            raise RuntimeError(
                "ConcurrentLatent base task requires 137-D observations and "
                "153-D critic states before expansion; got {} and {}".format(
                    self.num_obs, self.srl_full_obs_size
                )
            )
        self._latent_base_obs_buf = self.obs_buf
        self._latent_base_mirrored_obs_buf = self.obs_mirrored_buf
        self.num_observations = 153
        self.cfg["env"]["numObservations"] = self.num_observations
        self.obs_space = spaces.Box(
            np.full(self.num_observations, -np.inf, dtype=np.float32),
            np.full(self.num_observations, np.inf, dtype=np.float32),
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=torch.float32,
        )
        self.obs_mirrored_buf = torch.zeros_like(self.obs_buf)
        self._latent_observation_buffers_ready = True
        self._ensure_latent_body_masses()
        self.compute_observations()

    def _ensure_latent_body_masses(self):
        if self._latent_body_masses is not None:
            return
        if self.randomize and self._latent_track_randomized_body_masses:
            masses = []
            for env, handle in zip(self.envs, self.humanoid_handles):
                properties = self.gym.get_actor_rigid_body_properties(env, handle)
                masses.append([float(prop.mass) for prop in properties])
            self._latent_body_masses = torch.tensor(
                masses, device=self.device, dtype=torch.float32
            )
        else:
            properties = self.gym.get_actor_rigid_body_properties(
                self.envs[0], self.humanoid_handles[0]
            )
            nominal = torch.tensor(
                [float(prop.mass) for prop in properties],
                device=self.device,
                dtype=torch.float32,
            )
            self._latent_body_masses = nominal.unsqueeze(0).expand(
                self.num_envs, -1
            ).clone()

    def _latent_auxiliary_labels(self, env_ids=None):
        self._ensure_latent_body_masses()
        body_pos = self._rigid_body_pos if env_ids is None else self._rigid_body_pos[env_ids]
        root_rot = self.root_states[:, 3:7] if env_ids is None else self.root_states[env_ids, 3:7]
        contact_forces = self.contact_forces if env_ids is None else self.contact_forces[env_ids]

        mass = self._latent_body_masses if env_ids is None else self._latent_body_masses[env_ids]
        mass = mass.unsqueeze(-1)
        com_world = (body_pos * mass).sum(dim=1) / mass.sum(dim=1).clamp_min(1e-6)

        foot_pos = body_pos[:, self.feet_indices.long(), :]
        foot_force_z = contact_forces[:, self.feet_indices.long(), 2].clamp_min(0.0)
        # Height remains available when PhysX contact collection is disabled.
        height_contacts = (
            foot_pos[:, :, 2] <= self._latent_contact_height_threshold
        ).float()
        force_contacts = (foot_force_z > self._latent_contact_force_threshold).float()
        contacts = torch.maximum(height_contacts, force_contacts)
        force_sum = foot_force_z.sum(dim=1, keepdim=True)
        cop_weights = torch.where(
            force_sum > 1e-6, foot_force_z, contacts
        )
        weight_sum = cop_weights.sum(dim=1, keepdim=True)
        weighted_cop = (
            foot_pos * cop_weights.unsqueeze(-1)
        ).sum(dim=1) / weight_sum.clamp_min(1e-6)
        fallback_cop = foot_pos.mean(dim=1)
        cop_world = torch.where(weight_sum > 1e-6, weighted_cop, fallback_cop)
        com_minus_cop_local = quat_rotate_inverse(root_rot, com_world - cop_world)
        return com_minus_cop_local, contacts

    def compute_observations(self, env_ids=None):
        if not getattr(self, "_latent_observation_buffers_ready", False):
            return super().compute_observations(env_ids)

        expanded_obs = self.obs_buf
        expanded_mirrored = self.obs_mirrored_buf
        self.obs_buf = self._latent_base_obs_buf
        self.obs_mirrored_buf = self._latent_base_mirrored_obs_buf
        try:
            super().compute_observations(env_ids)
        finally:
            self.obs_buf = expanded_obs
            self.obs_mirrored_buf = expanded_mirrored

        com_cop, contacts = self._latent_auxiliary_labels(env_ids)
        count = self.num_envs if env_ids is None else len(env_ids)
        auxiliary = torch.zeros(
            (count, self.auxiliary_slot_dim),
            device=self.device,
            dtype=self.obs_buf.dtype,
        )
        auxiliary[:, : self.com_cop_dim] = com_cop
        auxiliary[:, self.com_cop_dim : self.com_cop_dim + self.contact_dim] = contacts

        mirrored = auxiliary.clone()
        mirrored[:, 1] *= -1.0
        mirrored[:, 3:5] = contacts[:, [1, 0]]
        if env_ids is None:
            self.obs_buf[:, :137] = self._latent_base_obs_buf
            self.obs_mirrored_buf[:, :137] = self._latent_base_mirrored_obs_buf
            self.obs_buf[:, -self.auxiliary_slot_dim :] = auxiliary
            self.obs_mirrored_buf[:, -self.auxiliary_slot_dim :] = mirrored
        else:
            self.obs_buf[env_ids, :137] = self._latent_base_obs_buf[env_ids]
            self.obs_mirrored_buf[env_ids, :137] = (
                self._latent_base_mirrored_obs_buf[env_ids]
            )
            self.obs_buf[env_ids, -self.auxiliary_slot_dim :] = auxiliary
            self.obs_mirrored_buf[env_ids, -self.auxiliary_slot_dim :] = mirrored
