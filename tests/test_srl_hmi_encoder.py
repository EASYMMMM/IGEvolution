"""CPU integration checks. Never creates a simulator or starts training.

Use the srlgym environment; optional SRL_STAGE4_CHECKPOINT selects a real ckpt.
The synthetic PPO test exercises a single optimizer update on fabricated data.
"""
import copy
import math
import os
import shlex
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import isaacgym  # noqa: F401 -- Isaac Gym must precede torch imports
import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.datasets import PPODataset

from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig
from isaacgymenvs.learning.SRLEvo.hmi_encoder.layout import SRL39Layout
from isaacgymenvs.learning.SRLEvo.hmi_encoder.runtime import EncoderRuntime, check_checkpoint_mode
from isaacgymenvs.learning.SRLEvo.hmi_encoder.training import update_srl
from isaacgymenvs.learning.SRLEvo.srl_models import ModelSRLContinuous
from isaacgymenvs.learning.SRLEvo.srl_network_builder import SRLBuilder, HumanoidBuilder, CentralCritic
from isaacgymenvs.learning.SRLEvo.srl_continuous_marl import SRL_MultiAgent
from isaacgymenvs.learning.SRLEvo.srl_players import SRLPlayerContinuous

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / 'isaacgymenvs/runs/SRL_Real_HRI_v2_stage4_HMI_27-12-15-43/nn/SRL_Real_HRI_v2_stage4_HMI_27-12-15-49.pth'
torch.set_num_threads(2)


def config(**kwargs):
    return EncoderConfig.from_dict(dict(enabled=True, warmup_rollouts=2, calibration_rollouts=1,
                                        ramp_rollouts=2, microbatch_size=7, **kwargs))


def make_model(human=False, srl_width=198):
    params = copy.deepcopy(yaml.safe_load((ROOT / 'isaacgymenvs/cfg/train/SRL_Real_HRIPPO.yaml').read_text())['params']['network'])
    if human:
        net = HumanoidBuilder.Network(params, input_shape=(117,), actions_num_humanoid=58, amp_input_shape=(210,))
    else:
        params['mlp']['units'] = params['mlp']['srl_units']
        params['mlp']['activation'] = params['mlp']['srl_activation']
        net = SRLBuilder.Network(params, input_shape=(srl_width,), actions_num_srl=6,
                                 priv_obs_num_srl=(srl_width + 12,), actor_command_mask_size=3)
    return ModelSRLContinuous.Network(net, role='humanoid' if human else 'srl',
                                     obs_shape=(117,) if human else (srl_width,), normalize_input=True,
                                     normalize_value=True, value_size=1)


def set_yaw(states, yaw):
    states[:, 3:7] = 0
    states[:, 5] = torch.sin(yaw / 2)
    states[:, 6] = torch.cos(yaw / 2)


def make_rollout(runtime, horizon=20):
    env = SimpleNamespace(_root_states=torch.zeros(runtime.num_envs, 13),
                          srl_root_states=torch.zeros(runtime.num_envs, 13))
    set_yaw(env.srl_root_states, torch.full((runtime.num_envs,), math.pi / 2))
    env._root_states[:, 7] = 1
    records, live = [], []
    runtime.begin_rollout(horizon)
    for t in range(horizon):
        obs = torch.randn(runtime.num_envs, 198) * 0.1
        obs[:, runtime.slots] = 0
        obs[:, 0] = t + 1
        mirror = obs.clone()
        mirror[:, 0] *= -1
        yaw = torch.full((runtime.num_envs,), t * 0.015)
        set_yaw(env._root_states, yaw)
        ids = torch.tensor([1]) if t == 10 and runtime.num_envs > 1 else torch.empty(0, dtype=torch.long)
        runtime.observe(obs, mirror, ids, step=t, env=env)
        records.append(obs)
        live.append(runtime.history.clone())
    runtime.finish_rollout()
    return torch.stack(records).transpose(0, 1).reshape(-1, 198), live


def make_agent(runtime, model=None):
    agent = object.__new__(SRL_MultiAgent)
    agent.hmi_encoder = runtime
    agent.model_srl = make_model() if model is None else model
    agent.model = make_model(human=True)
    agent.central_critic = CentralCritic(117 + agent.model_srl.a2c_network.actor_mlp[0].in_features,
                                        (1024, 512))
    agent.optimizer_srl = torch.optim.Adam(agent.model_srl.parameters(), lr=1e-4)
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=3e-5)
    agent.central_critic_opt = torch.optim.Adam(agent.central_critic.parameters(), lr=1e-4)
    agent._amp_input_mean_std = RunningMeanStd((210,))
    agent._normalize_amp_input = True
    agent.use_central_critic = True
    agent.ppo_device = 'cpu'
    agent.scaler = torch.cuda.amp.GradScaler(enabled=False)
    agent.scaler_srl = torch.cuda.amp.GradScaler(enabled=False)
    agent.central_critic_scaler = torch.cuda.amp.GradScaler(enabled=False)
    agent._preproc_obs = lambda x: x
    agent.ppo = True
    agent.mixed_precision = False
    agent.multi_gpu = False
    agent.mirror_loss = True
    agent.vec_env = SimpleNamespace(env=SimpleNamespace(mirror_act_srl_mat=torch.eye(6)))
    agent.e_clip = 0.2
    agent.clip_value = False
    agent.critic_coef = 0.
    agent.entropy_coef = 0.005
    agent.bounds_loss_coef = 0.01
    agent.sym_actor_loss_coef = 1.
    agent.truncate_grads = True
    agent.grad_norm = 1.
    agent._train_srl = True
    agent.last_lr = 1e-4
    agent.train_result = {}
    agent.update_counter = 0
    agent._joint_restored = False
    agent.epoch_num = 0
    agent.frame = 0
    agent.last_mean_rewards = 0.
    return agent


def make_batch(agent, obs):
    r = agent.hmi_encoder
    ids = torch.arange(len(obs))
    history, _, _ = r.sample(ids)
    agent.model_srl.eval()
    with torch.no_grad():
        payload, _ = r.payload(history)
        result = agent.model_srl({'obs': obs.clone(), 'priv_obs': torch.zeros(len(obs), 210),
                                  'is_train': False, 'actor_latent': payload})
    return dict(obs=obs, obs_mirrored=obs.clone(), priv_obses=torch.zeros(len(obs), 210),
                priv_obses_mirrored=torch.zeros(len(obs), 210), hmi_sample_ids=ids,
                actions=result['actions'], old_logp_actions=result['neglogpacs'],
                old_values=result['values'], returns=torch.ones(len(obs), 1),
                advantages=torch.linspace(-1, 1, len(obs)), mu=result['mus'], sigma=result['sigmas'])


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_features_causality_and_privileged_exclusion(self):
        r = EncoderRuntime(config(), 'cpu', 2, .015)
        obs = torch.randn(2, 198)
        first = SRL39Layout.features(obs)
        self.assertEqual(first.shape, (2, 41))
        obs[:, 39:] += 100  # old stacked frames and commands aren't current features
        obs[:, 32:35] += 100
        obs[:, 24:26] += 100
        torch.testing.assert_close(first, SRL39Layout.features(obs), rtol=0, atol=0)
        x = torch.randn(2, 64, 41)
        x[..., -1] = 1
        y = x.clone()
        y[:, 40:] += 10
        def features(data):
            return r.model.temporal(r.model.project(r.model.normalizer(data)).transpose(1, 2))
        torch.testing.assert_close(features(x)[..., :40], features(y)[..., :40], rtol=0, atol=0)
        r.model.normalizer.calibrate(x)
        torch.testing.assert_close(r.model.normalizer(torch.zeros_like(x)), torch.zeros_like(x))

    def test_rollout_windows_reset_future_frames_and_mirror(self):
        r = EncoderRuntime(config(), 'cpu', 2, .015)
        _, live = make_rollout(r)
        for env_id in range(2):
            for t in (0, 9, 10, 19):
                history, _, _ = r.sample(torch.tensor([env_id * 20 + t]))
                torch.testing.assert_close(history[0], live[t][env_id], rtol=0, atol=0)
        ids = torch.tensor([1, 21, 29, 30, 39])
        _, targets, masks = r.sample(ids)
        _, mirrored, mirrored_masks = r.sample(ids, mirrored=True)
        torch.testing.assert_close(targets[..., 0], torch.zeros_like(targets[..., 0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(targets[..., 1], -torch.ones_like(targets[..., 1]))
        self.assertTrue(masks[0, 2])  # t=1 +16 within episode and rollout
        self.assertFalse(masks[1, 2])  # env1 t=1 +16 crosses reset at t=10
        self.assertFalse(masks[3, 0])  # first frame after reset: yaw difference undefined
        self.assertFalse(masks[-1, 1]) # future beyond end
        torch.testing.assert_close(mirrored[..., 1:], -targets[..., 1:])
        torch.testing.assert_close(masks, mirrored_masks)
        # Prefix survives rollout boundary without borrowing other environments.
        r.finish_update()
        _, live2 = make_rollout(r)
        history, _, _ = r.sample(torch.tensor([0]))
        torch.testing.assert_close(history[0], live2[0][0])
        self.assertGreater(history[0, :-1, -1].sum(), 0)

    def test_real_stage4_actor_identity_and_critic_isolation(self):
        checkpoint_path = Path(os.environ.get('SRL_STAGE4_CHECKPOINT', str(DEFAULT_CKPT)))
        if not checkpoint_path.exists():
            self.skipTest('No stage4 checkpoint; set SRL_STAGE4_CHECKPOINT')
        runtime = EncoderRuntime(config(), 'cpu', 3, .015)
        agent = make_agent(runtime)
        agent._load_hsrl_checkpoint(str(checkpoint_path))
        self.assertEqual(runtime.alpha, 0.)
        self.assertEqual(runtime.completed_rollouts, 0)
        self.assertFalse(runtime.optimizer.state_dict()['state'])
        self.assertAlmostEqual(agent.optimizer.param_groups[0]['lr'], 3e-5)
        model = agent.model_srl.eval()
        self.assertEqual(model.a2c_network.actor_mlp[0].in_features, 198)
        obs = torch.randn(3, 198)
        priv = torch.randn(3, 210)
        obs[:, SRL39Layout.slots] = 0
        baseline = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False})
        r = EncoderRuntime(config(), 'cpu', 3, .015, training=False)
        r.observe(obs)
        payload, _ = r.payload()
        before = copy.deepcopy(model.state_dict())
        same = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False, 'actor_latent': payload})
        for key in ('mus', 'sigmas', 'values'):
            torch.testing.assert_close(baseline[key], same[key], rtol=0, atol=0)
        r.alpha = 1.
        changed = model({'obs': obs.clone(), 'priv_obs': priv, 'is_train': False, 'actor_latent': r.payload()[0]})
        self.assertFalse(torch.equal(changed['mus'], same['mus']))
        torch.testing.assert_close(changed['values'], same['values'], rtol=0, atol=0)
        obs[:, -3:] += 100
        command_changed = model({'obs': obs, 'priv_obs': priv, 'is_train': False, 'actor_latent': r.payload()[0]})
        torch.testing.assert_close(changed['mus'], command_changed['mus'], rtol=0, atol=0)
        self.assertEqual(before.keys(), model.state_dict().keys())
        model.zero_grad()
        r.model.zero_grad()
        actor = model({'obs': obs, 'priv_obs': priv, 'is_train': False, 'actor_latent': r.payload()[0]})
        actor['mus'].square().mean().backward()
        self.assertGreater(r.model.project.weight.grad.abs().sum().item(), 0.)  # PPO path, no auxiliary loss


    def test_ppo_microbatch_gradients_and_rms_freeze(self):
        r = EncoderRuntime(config(), 'cpu', 2, .015)
        obs, _ = make_rollout(r)
        r.alpha = .5
        agent = make_agent(r)
        batch = make_batch(agent, obs)
        previous_actor = agent.model_srl.a2c_network.actor_mlp[0].weight.detach().clone()
        previous_encoder = r.model.project.weight.detach().clone()
        previous_rms = copy.deepcopy(agent.model_srl.running_mean_std.state_dict())
        # Same update, once with a full minibatch and once using small microbatches.
        reference = make_agent(EncoderRuntime(config(), 'cpu', 2, .015))
        reference.model_srl.load_state_dict(agent.model_srl.state_dict())
        reference.hmi_encoder = copy.deepcopy(r)
        reference.hmi_encoder.config.microbatch_size = len(obs)
        update_srl(agent, batch)
        update_srl(reference, batch)
        self.assertFalse(torch.equal(previous_actor, agent.model_srl.a2c_network.actor_mlp[0].weight))
        self.assertFalse(torch.equal(previous_encoder, r.model.project.weight))
        for key, value in previous_rms.items():
            torch.testing.assert_close(value, agent.model_srl.running_mean_std.state_dict()[key], rtol=0, atol=0)
        for name, value in agent.model_srl.named_parameters():
            self.assertTrue(torch.isfinite(value).all())
            torch.testing.assert_close(value, dict(reference.model_srl.named_parameters())[name], atol=2e-5, rtol=2e-4)
        for key in ('hmi_loss_0', 'hmi_loss_8', 'hmi_loss_16', 'actor_loss_srl'):
            torch.testing.assert_close(agent.train_result[key], reference.train_result[key], atol=2e-6, rtol=2e-5)
        # Calibration/curriculum changes only after the complete PPO update.
        old_alpha = r.alpha
        r.finish_update()
        self.assertEqual(old_alpha, r.alpha)
        self.assertGreater(r.model.normalizer.count, 0)
        # Exercise the real logger's expected keys (including disabled critic symmetry).
        agent._train_humanoid = False
        agent._srl_teacher_checkpoint = False
        agent.writer = SimpleNamespace(add_scalar=lambda *args: None)
        info = {key: [value] for key, value in agent.train_result.items()}
        info.update(update_time=0., play_time=0., total_time=0., prepare_dataset_time=0.)
        agent._log_train_info(info, 0)

    def test_joint_checkpoint_save_resume_and_player(self):
        r = EncoderRuntime(config(), 'cpu', 2, .015)
        obs, _ = make_rollout(r)
        r.alpha = .5
        r.completed_rollouts = 3
        agent = make_agent(r)
        update_srl(agent, make_batch(agent, obs))
        agent.update_counter = 17
        agent.epoch_num = 9
        agent.frame = 1234
        agent.get_full_state_weights = lambda: dict(model=agent.model.state_dict(),
            optimizer=agent.optimizer.state_dict(), amp_input_mean_std=agent._amp_input_mean_std.state_dict(),
            epoch=agent.epoch_num, frame=agent.frame, last_mean_rewards=agent.last_mean_rewards)
        with tempfile.TemporaryDirectory() as directory:
            fn = str(Path(directory) / 'encoder')
            agent.save(fn)
            state = torch.load(fn + '.pth', map_location='cpu', weights_only=False)
            restored = make_agent(EncoderRuntime(config(), 'cpu', 2, .015))
            restored.restore(fn + '.pth')
            self.assertEqual(restored.update_counter, 17)
            self.assertEqual(restored.epoch_num, 9)
            self.assertEqual(restored._start_frame, 1234)
            self.assertEqual(restored.hmi_encoder.alpha, .5)
            self.assertEqual(restored.hmi_encoder.completed_rollouts, 3)
            self.assertAlmostEqual(restored.optimizer.param_groups[0]['lr'], 3e-5)
            self.assertTrue(restored.hmi_encoder.optimizer.state_dict()['state'])
            # Restoring an episode clears history; replay the same history for parity.
            history = r.history.clone()
            restored.hmi_encoder.history.copy_(history)
            agent.model_srl.eval(); restored.model_srl.eval()
            def action(model, runtime):
                return model({'obs': obs[:2].clone(), 'priv_obs': torch.zeros(2, 210),
                              'is_train': False, 'actor_latent': runtime.payload()[0]})['mus']
            torch.testing.assert_close(action(agent.model_srl, r), action(restored.model_srl, restored.hmi_encoder), rtol=0, atol=0)
            # Exercise the actual player's restore and get_action, without a simulator.
            env = SimpleNamespace(num_envs=2, control_dt=.015, srl_hmi_control=True,
                cfg={'env': {'srl_policy_obs_remove_ids': [0, 1, 2, 3], 'srl_obs_frame_stack': 5}},
                get_srl_obs_size=lambda: 198)
            player = object.__new__(SRLPlayerContinuous)
            player.hmi_encoder = EncoderRuntime(config(), 'cpu', 2, .015, training=False)
            player.env = env
            player.device = 'cpu'
            player.model = make_model(human=True).eval()
            player.model_srl = make_model().eval()
            player.normalize_input = True
            player._normalize_amp_input = True
            player._amp_input_mean_std = RunningMeanStd((210,))
            player.restore(fn + '.pth')
            player.hmi_encoder.history.copy_(history)
            torch.testing.assert_close(action(player.model_srl, player.hmi_encoder), action(agent.model_srl, r), rtol=0, atol=0)
            player.has_batch_dimension = True
            player.obs_num_humanoid, player.obs_num_srl, player.priv_obs_num_srl = 117, 198, 210
            player.states = None
            player.clip_actions = False
            player._preproc_obs = lambda x: x
            current = obs[:2].clone()
            r.observe(current)
            expected = action(agent.model_srl, r)
            extra = torch.arange(24, dtype=current.dtype).reshape(2, 12)
            raw = {'obs': torch.cat((torch.zeros(2, 117), current), -1),
                   'srl_priv_extra_obs': extra}
            player.env.reset_done = lambda: (raw, torch.empty(0, dtype=torch.long))
            player_obs, _ = player._env_reset_done()
            critic_inputs = []
            hook = player.model_srl.a2c_network.critic_mlp.register_forward_pre_hook(
                lambda module, args: critic_inputs.append(args[0].clone()))
            try:
                outputs = player.get_action(player_obs, True)
            finally:
                hook.remove()
            torch.testing.assert_close(critic_inputs[0], torch.cat((extra, current), -1))
            torch.testing.assert_close(outputs[:, -6:], expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'refusing'):
                check_checkpoint_mode(None, state)
            wrong = EncoderRuntime(config(channels=16), 'cpu', 2, .015)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                wrong.load(state['hmi_encoder'])
            with self.assertRaisesRegex(ValueError, 'period'):
                EncoderRuntime(config(), 'cpu', 2, .03).load(state['hmi_encoder'])

    def test_candidate_seed_keeps_encoder_and_policy_but_starts_new_training(self):
        source = make_agent(EncoderRuntime(config(), 'cpu', 2, .015))
        source.hmi_encoder.alpha = 1.
        source.hmi_encoder.completed_rollouts = 700
        source.model_srl.running_mean_std.running_mean.normal_()
        source.model_srl.running_mean_std.running_var.uniform_(1., 2.)
        state = dict(model=source.model.state_dict(), model_srl=source.model_srl.state_dict(),
                     amp_input_mean_std=source._amp_input_mean_std.state_dict(),
                     central_critic=source.central_critic.state_dict(),
                     hmi_encoder=source.hmi_encoder.checkpoint(), epoch=700, frame=9000)
        # A policy-only seed has no optimizer or scaler state. Full resume is tested separately.
        state['hmi_encoder']['optimizer'] = None
        obs = torch.randn(2, 198)
        design_obs = torch.cat((obs[:, :-3], torch.randn(2, 7), obs[:, -3:]), dim=-1)
        human = torch.randn(2, 117)
        privileged = torch.randn(2, 12)

        def outputs(agent, observations):
            agent.model_srl.eval()
            agent.hmi_encoder.observe(observations)
            with torch.no_grad():
                policy = agent.model_srl(dict(obs=observations, is_train=False,
                    priv_obs=torch.cat((privileged, observations), dim=-1),
                    actor_latent=agent.hmi_encoder.payload()[0]))
                joint = torch.cat((human, agent.model_srl.norm_obs(observations)), dim=-1)
                return (policy['mus'], policy['values'], *agent.central_critic(joint))

        expected = outputs(source, obs)
        for observations in (obs, design_obs):
            width = observations.shape[-1]
            with self.subTest(width=width):
                target = make_agent(EncoderRuntime(config(), 'cpu', 2, .015), make_model(srl_width=width))
                target._start_frame = 1
                target._load_joint_checkpoint(state)
                self.assertEqual((target.epoch_num, target._start_frame, target.update_counter), (0, 1, 0))
                self.assertFalse(target._joint_restored)
                self.assertEqual((target.hmi_encoder.alpha, target.hmi_encoder.completed_rollouts), (1., 700))
                for optimizer in (target.optimizer, target.optimizer_srl, target.central_critic_opt,
                                  target.hmi_encoder.optimizer):
                    self.assertFalse(optimizer.state)
                actual = outputs(target, observations)
                for old, new in zip(expected, actual):
                    torch.testing.assert_close(old, new)
        self.assertTrue(torch.all(target.model_srl.running_mean_std.running_var[-10:-3] == 1.))
        self.assertTrue(torch.all(target.model_srl.running_mean_std.running_mean[-10:-3] == 0.))
        env = SimpleNamespace(srl_hmi_control=True, get_srl_obs_size=lambda: 205,
                              cfg={'env': {'design_param_obs': True, 'srl_policy_obs_remove_ids': [0, 1, 2, 3]}})
        SRL39Layout.validate_env(env)
        env.cfg['env']['design_param_obs'] = False
        with self.assertRaisesRegex(ValueError, 'expected 198'):
            SRL39Layout.validate_env(env)
        with self.assertRaisesRegex(ValueError, 'hsrl_checkpoint'):
            target._load_joint_checkpoint(state, resume=True)

    def test_bash_training_command_composes_without_launch(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        script = (ROOT / 'srl_real_hri.bash').read_text()
        extension = script[script.index('# 第五阶段训练:'):]
        command = extension[extension.index('python SRL_Evo_train.py'):].split('\n\n')[0]
        args = shlex.split(command.replace('\\\n', ' '))[2:]
        launcher = (ROOT / 'run_srl_hmi_stage5.sh').read_text()
        launcher_args = shlex.split(launcher.split('SRL_Evo_train.py ', 1)[1].replace('\\\n', ' '))
        self.assertEqual(launcher_args[-1], '$@')
        self.assertEqual(launcher_args[:-1], args)
        with initialize_config_dir(config_dir=str(ROOT / 'isaacgymenvs/cfg'), version_base='1.1'):
            composed = compose(config_name='config', overrides=args)
            settings = OmegaConf.to_container(composed, resolve=True)
        train = settings['train']['params']['config']
        cfg = EncoderConfig.from_dict(train['hmi_encoder'])
        self.assertTrue(cfg.enabled)
        self.assertTrue(train['train_humanoid'])
        self.assertEqual(train['train_humanoid_freq'], 10)
        self.assertAlmostEqual(train['learning_rate'] * cfg.humanoid_lr_scale, 3e-5)
        self.assertEqual(ROOT / 'isaacgymenvs' / train['hsrl_checkpoint'], DEFAULT_CKPT)
        self.assertEqual(settings['task']['env']['srl_obs_frame_stack'], 5)
        self.assertTrue(settings['task']['env']['srl_hmi_control'])

    def test_gate_schedule_has_exact_warmup_and_ramp_lengths(self):
        r = EncoderRuntime(config(), 'cpu', 1, .015)
        for completed, expected in ((0, 0.), (1, 0.), (2, .5), (3, 1.), (4, 1.)):
            r.completed_rollouts = completed
            r.begin_rollout(20)
            self.assertEqual(r.alpha, expected)

    def test_config_layout_and_dataset_ids(self):
        with self.assertRaises(ValueError):
            EncoderConfig.from_dict({'latent_dim': 16})
        with self.assertRaises(ValueError):
            EncoderConfig.from_dict({'unknown_setting': 1})
        r = EncoderRuntime(config(), 'cpu', 2, .015)
        obs, live = make_rollout(r)
        data = PPODataset(40, 8, False, False, 'cpu', 4)
        data.update_values_dict({'obs': obs, 'hmi_sample_ids': torch.arange(40)})
        for i in range(len(data)):
            batch = data[i]
            history, _, _ = r.sample(batch['hmi_sample_ids'])
            torch.testing.assert_close(history[:, -1], SRL39Layout.features(batch['obs']))


if __name__ == '__main__':
    unittest.main()
