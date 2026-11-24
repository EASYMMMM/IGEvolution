"""HMI input isolation and stage-3 compatibility checks (srlgym environment)."""
import copy
import math
from pathlib import Path
import unittest

import isaacgym  # Must precede torch for the simulator bindings.
import torch
import yaml

from isaacgymenvs.learning.SRLEvo.srl_models import ModelSRLContinuous
from isaacgymenvs.learning.SRLEvo.srl_network_builder import SRLBuilder
from isaacgymenvs.tasks.SRLEvo.srl_real_hri_base import SRL_Real_HRI_Base

ROOT = Path(__file__).resolve().parents[1]


class HMIControlTests(unittest.TestCase):
    def test_actor_mask_after_normalization_preserves_critic_and_checkpoint(self):
        torch.manual_seed(7)
        params = yaml.safe_load((ROOT / 'isaacgymenvs/cfg/train/SRL_Real_HRIPPO.yaml').read_text())['params']['network']
        params = copy.deepcopy(params)
        params['mlp']['units'] = params['mlp']['srl_units']
        params['mlp']['activation'] = params['mlp']['srl_activation']
        net = SRLBuilder.Network(params, input_shape=(198,), actions_num_srl=6,
                                 priv_obs_num_srl=(210,), actor_command_mask_size=3)
        model = ModelSRLContinuous.Network(net, role='srl', obs_shape=(198,),
                                          normalize_input=True, normalize_value=True, value_size=1)
        checkpoint = ROOT / 'isaacgymenvs/runs/SRL_Real_HRI_v2_stage3_19-20-10-40/nn/SRL_Real_HRI_v2_stage3_19-20-10-46.pth'
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_srl'], strict=True)
        model.eval()
        model.running_mean_std.running_mean[-3:] = 2.0
        observed = {}

        def capture(name):
            def hook(module, args):
                observed[name] = args[0].detach().clone()
            return hook

        net.actor_mlp.register_forward_pre_hook(capture('actor'))
        net.critic_mlp.register_forward_pre_hook(capture('critic'))
        obs = torch.randn(4, 198)
        priv = torch.cat([torch.randn(4, 12), obs], dim=-1)
        original = obs.clone()
        first = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False})
        torch.testing.assert_close(observed['actor'][:, -3:], torch.zeros(4, 3))
        torch.testing.assert_close(observed['critic'], priv)
        torch.testing.assert_close(obs, original)
        normalized = model.norm_obs(obs.clone())
        self.assertGreater(normalized[:, -3:].abs().sum().item(), 0)
        obs[:, -3:] += 3.0
        priv[:, -3:] = obs[:, -3:]
        second = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False})
        torch.testing.assert_close(first['mus'], second['mus'], rtol=0, atol=0)
        torch.testing.assert_close(observed['critic'][:, -3:], obs[:, -3:])
        # HMI is still an active input, and the ordinary/teacher mode is unmasked.
        obs[:, 26:39] += 1.0
        third = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False})
        self.assertFalse(torch.equal(second['mus'], third['mus']))
        keys = set(model.state_dict())
        net.actor_command_mask_size = 0
        model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False})
        torch.testing.assert_close(observed['actor'], model.norm_obs(obs.clone()))
        self.assertEqual(keys, set(model.state_dict()))

    def test_heading_frames_preserve_reward_teacher_commands_and_partial_reset(self):
        # Use the real observation assembly with synthetic physical states, so
        # all-environment and partial-reset frame stacking can be checked on CPU.
        env = object.__new__(SRL_Real_HRI_Base)
        env.num_environments = 2
        env.srl_hmi_control = True
        env._srl_policy_yaw_id = 3  # full frame index 7, after removing indices 0:4
        env.srl_policy_obs_ids = torch.arange(4, 43)
        env.teacher_srl_policy_obs_ids = torch.arange(4, 30)
        env.teacher_srl_full_obs_num = 30
        env._root_states = torch.zeros(2, 13)
        env.srl_root_states = torch.zeros(2, 13)
        for state, degrees in ((env._root_states, [179., 30.]), (env.srl_root_states, [-179., 0.])):
            radians = torch.tensor(degrees) * math.pi / 180
            state[:, 5] = torch.sin(radians / 2)
            state[:, 6] = torch.cos(radians / 2)
        for name, width in [('srl_obs', 39), ('srl_obs_mirrored', 39), ('srl_full_obs', 43), ('teacher_srl_obs', 26)]:
            setattr(env, name + '_buffer', torch.zeros(2, 5, width))
            setattr(env, name + '_buf', torch.zeros(2, width * 5 + 3))
        env.obs_buf = torch.zeros(2, 315)
        env.srl_priv_extra_obs_buf = torch.zeros(2, 12)
        env.srl_priv_extra_mirrored_obs_buf = torch.zeros(2, 12)
        env.target_vel_x = torch.tensor([1., 0.7])
        env.target_ang_vel_z = torch.tensor([0.4, -0.5])
        env.target_pelvis_height = torch.tensor([0.95, 0.9])
        env.potentials = torch.zeros(2)
        env.prev_potentials = torch.zeros(2)
        full = torch.ones(2, 43)
        full[:, 7] = torch.tensor([0.8, -0.9])  # Deliberately unrelated trajectory heading.
        humanoid = torch.randn(2, 117)
        source = (humanoid, full, full.clone(), torch.ones(2, 12), torch.ones(2, 12),
                  full[:, :30].clone(), torch.zeros(2), torch.zeros(2))
        env._compute_env_obs = lambda ids: tuple(x.clone() if ids is None else x[ids].clone() for x in source)
        env._compute_observations()
        expected = torch.tensor([-2., 30.]) * math.pi / 180
        torch.testing.assert_close(env.srl_obs_buffer[:, :, 3], expected[:, None].expand(-1, 5), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(env.srl_obs_mirrored_buffer[:, :, 3], -env.srl_obs_buffer[:, :, 3])
        torch.testing.assert_close(env.srl_full_obs_buffer[:, :, 7], full[:, 7:8].expand(-1, 5))
        torch.testing.assert_close(env.teacher_srl_obs_buffer[:, :, 3], full[:, 7:8].expand(-1, 5))
        commands = torch.stack([env.target_vel_x, env.target_ang_vel_z, env.target_pelvis_height], dim=-1)
        for buf in (env.srl_obs_buf, env.srl_full_obs_buf, env.teacher_srl_obs_buf):
            torch.testing.assert_close(buf[:, -3:], commands)
        torch.testing.assert_close(env.obs_buf[:, :117], humanoid)
        previous = env.srl_obs_buf[0].clone()
        env._root_states[1, 5:7] = torch.tensor([0., 1.])
        env._compute_observations(torch.tensor([1]))
        torch.testing.assert_close(env.srl_obs_buf[0], previous)
        self.assertEqual(env.srl_obs_buffer[1, 0, 3].item(), 0.)
        torch.testing.assert_close(env.srl_obs_buffer[1, 1:, 3], expected[1].expand(4))


if __name__ == '__main__':
    unittest.main()
