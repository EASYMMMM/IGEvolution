"""The ONLY place that knows the legacy observation ordering and simulator fields.

To adopt a different observation layout, provide a new version of this adapter;
network, rollout storage, and PPO code operate on features rather than indices.
"""
import torch


class SRL39Layout:
    name = 'srl39_zero_moments_v1'
    obs_dim = 198
    design_dim = 7
    command_dim = 3
    frame_dim = 39
    feature_dim = 41
    slots = tuple(39 * frame + i for frame in range(5) for i in (32, 33, 34))

    @classmethod
    def validate_env(cls, env):
        has_design = env.cfg['env'].get('design_param_obs', False)
        expected = cls.obs_dim + (cls.design_dim if has_design else 0)
        if env.get_srl_obs_size() != expected:
            raise ValueError(f'HMI encoder expected {expected} SRL observations for this configuration')
        if not env.srl_hmi_control or not env.cfg['env'].get('HMI_obs_enable', True):
            raise ValueError('Enable task.env.srl_hmi_control and HMI_obs_enable for the encoder')
        if list(env.cfg['env'].get('srl_policy_obs_remove_ids', [])) != [0, 1, 2, 3]:
            raise ValueError('Unexpected SRL frame ordering')
        if env.cfg['env'].get('srl_obs_frame_stack', 5) != 5:
            raise ValueError('HMI encoder layout requires five legacy frames')

    @classmethod
    def features(cls, obs):
        if obs.shape[-1] not in (cls.obs_dim, cls.obs_dim + cls.design_dim):
            raise ValueError('Expected five 39D frames, optional seven morphology values, and three commands')
        frame = obs[..., :cls.frame_dim]
        # Six periodic angles: relative human/robot orientation and base error.
        angles = torch.cat((frame[..., 26:29], frame[..., 3:6]), dim=-1)
        return torch.cat((angles.sin(), angles.cos(), frame[..., 29:32],
                          frame[..., 35:39], frame[..., 0:3], frame[..., 6:24],
                          torch.ones_like(frame[..., :1])), dim=-1)

    @staticmethod
    def yaw(quaternion):
        x, y, z, w = quaternion.unbind(-1)  # Isaac Gym xyzw
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @classmethod
    def label_state(cls, env):
        # Privileged simulator state is used ONLY as a detached training label.
        human = env._root_states.detach()
        robot = env.srl_root_states.detach()
        return human[:, 7:9], cls.yaw(human[:, 3:7]), cls.yaw(robot[:, 3:7])


def inject_latent(actor_obs, payload):
    """Called after legacy normalization/masking; never changes critic input."""
    latent, alpha, slots = payload
    result = actor_obs.clone()
    result[..., slots] = (1.0 - alpha) * actor_obs[..., slots] + alpha * latent
    return result
