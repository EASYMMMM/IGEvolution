"""Frozen stage-5 encoder evaluation. No training and no modification of checkpoints.

Run from any directory with the srlgym Python. See manifest.json in the output.
Simulation runs use deterministic policy means and score only the first episode
of each environment; subsequent resets are simulated but excluded from scores.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaacgym  # must precede torch
import numpy as np
import torch
from omegaconf import OmegaConf
import isaacgymenvs
from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig
from isaacgymenvs.learning.SRLEvo.hmi_encoder.layout import SRL39Layout, inject_latent
from isaacgymenvs.learning.SRLEvo.hmi_encoder.runtime import EncoderRuntime
from isaacgymenvs.learning.SRLEvo.srl_models import ModelSRLContinuous
from isaacgymenvs.learning.SRLEvo.srl_network_builder import HumanoidBuilder, SRLBuilder

RUN = ROOT / 'isaacgymenvs/runs/SRL_Real_HRI_v2_stage5_HMIEncoder_27-19-38-49'
CKPT = RUN / 'nn/SRL_Real_HRI_v2_stage5_HMIEncoder_27-19-38-56_best.pth'
CONDITIONS = ('intact', 'zero', 'mean', 'shuffle', 'delay16', 'human_swap', 'short5')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha_array(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def load_policy(device='cuda:0'):
    cfg = OmegaConf.load(RUN / 'config.yaml')
    params = OmegaConf.to_container(cfg.train.params.network, resolve=True)
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    models = []
    for human in (True, False):
        p = copy.deepcopy(params)
        if human:
            net = HumanoidBuilder.Network(p, input_shape=(117,), actions_num_humanoid=58,
                                          amp_input_shape=(210,))
        else:
            p['mlp']['units'] = p['mlp']['srl_units']
            p['mlp']['activation'] = p['mlp']['srl_activation']
            net = SRLBuilder.Network(p, input_shape=(198,), actions_num_srl=6,
                                     priv_obs_num_srl=(210,), actor_command_mask_size=3)
        m = ModelSRLContinuous.Network(net, role='humanoid' if human else 'srl',
                                      obs_shape=(117,) if human else (198,),
                                      normalize_input=True, normalize_value=True, value_size=1)
        m.load_state_dict(checkpoint['model' if human else 'model_srl'], strict=True)
        m.to(device).eval()
        for param in m.parameters():
            param.requires_grad_(False)
        models.append(m)
    return cfg, checkpoint, models


def actor(model, obs, latent=None, alpha=1.0):
    x = model.norm_obs(obs)
    net = model.a2c_network
    if model.role == 'srl':
        x = x.clone()
        x[..., -3:] = 0
        if latent is not None:
            x = inject_latent(x, (latent, alpha, SRL39Layout.slots))
    return net.mu_act(net.mu(net.actor_mlp(x)))


def main():
    from isaacgym import gymtorch
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS + ('suite',), default='intact')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--randomize-layout', action='store_true',
                        help='Randomly assign paired conditions to physical environment IDs.')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = list(CONDITIONS) + ['intact_repeat'] if args.condition == 'suite' else [args.condition]
    group_size = args.num_envs
    total_envs = group_size * len(conditions)
    os.chdir(ROOT / 'isaacgymenvs')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(args.seed)
    cfg, checkpoint, (human, robot) = load_policy()
    cfg.num_envs = total_envs
    cfg.task.env.numEnvs = total_envs
    cfg.seed = args.seed
    cfg.test = True
    cfg.headless = True
    cfg.force_render = False
    cfg.wandb_activate = False
    # Keep the original 5000-step task duration even in short smoke/calibration runs.
    cfg.task.env.episodeLength = 5000
    cfg.task.env.cameraFollow = False
    cfg.task.env.enableDebugVis = False
    cfg.sim_device = cfg.rl_device = 'cuda:0'
    cfg.graphics_device_id = -1
    env = isaacgymenvs.make(args.seed, 'SRL_Real_HRI', total_envs,
                            'cuda:0', 'cuda:0', -1, True, False, False, False, cfg)
    runtime = EncoderRuntime(EncoderConfig.from_dict(checkpoint['hmi_encoder']['config']),
                             'cuda:0', total_envs, env.control_dt, training=False)
    runtime.load(checkpoint['hmi_encoder'], resume_optimizer=False)
    runtime.model.eval()
    for param in runtime.model.parameters():
        param.requires_grad_(False)
    order = torch.arange(total_envs, device='cuda:0')
    if args.randomize_layout:
        order = torch.as_tensor(np.random.default_rng(args.seed+91000).permutation(total_envs), device='cuda:0')
    inverse = torch.argsort(order)
    # Each condition receives the same random trajectory and phase draw.
    # Simulate groups together for speed; collision groups keep environments independent.
    for group in range(len(conditions)):
        seed_all(args.seed)
        env.reset_idx(order[group*group_size:(group+1)*group_size])
    # PhysX tensor setters must be committed once for the entire batch between
    # simulations. Repeated subset reset calls above generate matched tasks,
    # but their state-buffer refreshes can overwrite earlier groups' DOFs.
    # Re-establish and submit every initial state in one full-batch operation.
    env._dof_pos[:] = env._initial_dof_pos
    env._dof_vel[:] = env._initial_dof_vel
    env._root_states[:] = env._initial_root_states
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
    env._reset_srl_action_filter(torch.arange(total_envs, device='cuda:0'))
    env._compute_observations()
    observations, reset_ids = env.reset_done()
    runtime.reset(inverse[reset_ids])
    initial = torch.cat((env._root_states, env._dof_pos, env.phase_buf[:, None]), 1).clone()
    initial[:, :2] -= env._initial_root_states[:, :2]
    trajectory = env._traj_gen._traj_pos - env._traj_gen._traj_pos[:, :1]
    initial = initial[order]
    trajectory = trajectory[order]
    initial_hashes, trajectory_hashes = [], []
    for group in range(len(conditions)):
        sl = slice(group*group_size, (group+1)*group_size)
        torch.testing.assert_close(initial[sl], initial[:group_size], atol=1e-5, rtol=0)
        torch.testing.assert_close(trajectory[sl], trajectory[:group_size], atol=1e-4, rtol=0)
        initial_hashes.append(sha_array(initial[sl]))
        trajectory_hashes.append(sha_array(trajectory[sl]))
    prefix = ('calibration' if args.calibration else args.condition) + f'_seed{args.seed}'
    donor = None
    if args.condition in ('shuffle', 'human_swap'):
        donor = np.load(args.output / f'intact_seed{args.seed}_data.npz')
    donor_ids = torch.roll(torch.arange(group_size, device='cuda:0'), 1)
    mean_z = None
    if 'mean' in conditions:
        mean_z = torch.as_tensor(np.load(args.output / 'calibration_mean.npy'), device='cuda:0')
    active = torch.ones(total_envs, device='cuda:0', dtype=torch.bool)
    totals = torch.zeros(total_envs, 8, device='cuda:0', dtype=torch.float64)
    # columns: steps, position squared error, velocity squared error, force,
    # mechanical abs power, action delta squared, SRL reward, muscle activation squared.
    end_steps = torch.full((total_envs,), args.steps, device='cuda:0', dtype=torch.long)
    terminated = torch.zeros(total_envs, device='cuda:0', dtype=torch.bool)
    previous_action = torch.zeros(total_envs, 6, device='cuda:0')
    delay = torch.zeros(17, total_envs, 15, device='cuda:0')
    first = True
    repeat_first_step_action_max_diff = None
    records = {k: [] for k in ('features', 'states', 'active', 'episode', 'z', 'prediction', 'obs',
                                'sample_steps', 'trace_time', 'trace_survival', 'trace_position')}
    calibration_z = []
    started = time.monotonic()
    with torch.no_grad():
        for step in range(args.steps):
            if step:
                observations, reset_ids = env.reset_done()
                runtime.reset(inverse[reset_ids])
                delay[:, inverse[reset_ids]] = 0
            raw = observations['obs'][order]
            hobs, robs = raw[:, :117], raw[:, -198:]
            runtime.observe(robs)
            z, pred = runtime.model(runtime.history)
            chosen = z.clone()
            if args.condition == 'suite':
                delay[step % 17] = z
                for group, condition in enumerate(conditions):
                    sl = slice(group*group_size, (group+1)*group_size)
                    if condition == 'zero':
                        chosen[sl] = 0
                    elif condition == 'mean':
                        chosen[sl] = mean_z
                    elif condition == 'shuffle':
                        chosen[sl] = z[donor_ids]
                    elif condition == 'delay16':
                        chosen[sl] = delay[(step-16) % 17, sl] if step >= 16 else 0
                    elif condition == 'short5':
                        short = runtime.history[sl].clone()
                        short[:, :-5] = 0
                        chosen[sl] = runtime.model(short)[0]
                    elif condition == 'human_swap':
                        altered = runtime.history[sl].clone()
                        altered[..., :19] = runtime.history[donor_ids, :, :19]
                        altered[..., :19] *= altered[..., -1:]
                        chosen[sl] = runtime.model(altered)[0]
            elif args.condition == 'zero':
                chosen = torch.zeros_like(z)
            elif args.condition == 'mean':
                chosen = mean_z.expand_as(z)
            elif args.condition == 'shuffle':
                chosen = torch.as_tensor(donor['z_all'][step], device='cuda:0')[donor_ids]
            elif args.condition == 'delay16':
                delay[step % 17] = z
                chosen = delay[(step - 16) % 17].clone() if step >= 16 else torch.zeros_like(z)
            elif args.condition == 'short5':
                short = runtime.history.clone()
                short[:, :-5] = 0
                chosen = runtime.model(short)[0]
            elif args.condition == 'human_swap':
                lo = max(0, step - 63)
                dh = torch.zeros_like(runtime.history)
                source = torch.as_tensor(donor['features'][lo:step+1], device='cuda:0').transpose(0, 1)
                dep = torch.as_tensor(donor['episode'][lo:step+1], device='cuda:0').transpose(0, 1)
                source = source * (dep == dep[:, -1:])[..., None]
                dh[:, -source.shape[1]:] = source
                altered = runtime.history.clone()
                altered[..., :19] = dh[donor_ids, :, :19]
                altered[..., :19] *= altered[..., -1:]
                chosen = runtime.model(altered)[0]
            mu_h = actor(human, hobs)
            mu_r = actor(robot, robs, chosen)
            if args.condition == 'suite' and step == 1:
                repeat_first_step_action_max_diff = max(
                    float((mu_h[:group_size]-mu_h[-group_size:]).abs().max()),
                    float((mu_r[:group_size]-mu_r[-group_size:]).abs().max()))
                if repeat_first_step_action_max_diff > .01:
                    raise RuntimeError('Repeated intact control diverged immediately; audit simulator initialization.')
            if args.diagnostic and args.condition == 'suite' and step in (0,1,2,5,10,20,49):
                b=slice(0,group_size);r=slice(-group_size,None)
                print('REPEAT_DIAGNOSTIC',step,{k:float((x[b]-x[r]).abs().max()) for k,x in
                      [('human_obs',hobs),('robot_obs',robs),('latent',z),('human_action',mu_h),('robot_action',mu_r)]},flush=True)
            if first:
                # Independently verify the fast inference path against the official wrapper.
                official = robot({'is_train': False, 'obs': robs.clone(),
                                  'priv_obs': torch.zeros(total_envs, 210, device='cuda:0'),
                                  'actor_latent': (chosen, 1.0, runtime.slots)})['mus']
                torch.testing.assert_close(mu_r, official, atol=1e-6, rtol=1e-6)
                first = False
            action = torch.cat((mu_h, mu_r), 1).clamp(-1, 1)
            if not torch.isfinite(action).all():
                raise RuntimeError(f'Nonfinite action at {step}')
            if args.collect:
                vel, yaw, robot_yaw = SRL39Layout.label_state(env)
                vel, yaw, robot_yaw = vel[order], yaw[order], robot_yaw[order]
                records['features'].append(runtime.history[:group_size, -1].cpu().numpy().copy())
                records['states'].append(torch.cat((vel, yaw[:, None], robot_yaw[:, None]), 1)[:group_size].cpu().numpy())
                records['active'].append(active[:group_size].cpu().numpy().copy())
                records['episode'].append(runtime.episode[:group_size].cpu().numpy().copy())
                records['z'].append(z[:group_size].cpu().numpy())
                if step % 8 == 0:
                    records['obs'].append(robs[:group_size].cpu().numpy().copy())
                    records['prediction'].append(pred[:group_size].cpu().numpy())
                    records['sample_steps'].append(step)
            if args.calibration and step >= 64 and step % 8 == 0:
                calibration_z.append(z[active].cpu().numpy())
            _, _, done, info = env.step(action[inverse])
            done = done[order]
            ids = torch.arange(total_envs, device='cuda:0')
            now = env.progress_buf * env.control_dt
            target = env._traj_gen.get_position(ids, now)
            future = env._traj_gen.get_observation_points(ids, now, [0.5])
            target_vel = (future[:, 0, :2] - target[:, :2]) / 0.5
            p2 = ((target[:, :2] - env._root_states[:, :2]) ** 2).sum(-1)[order]
            v2 = ((target_vel - env._root_states[:, 7:9]) ** 2).sum(-1)[order]
            force = env.virtual_load_cell_buf[:, :3].norm(dim=-1)[order]
            power = (env.srl_applied_torques * env._dof_vel[:, -6:]).abs().sum(-1)[order]
            jerk = ((action[:, -6:] - previous_action) ** 2).mean(-1)
            activation = env.muscle_actuator.a.square().mean(-1)[order]
            values = torch.stack((torch.ones_like(p2), p2, v2, force, power, jerk,
                                  info['srl_rewards'][order], activation), 1)
            totals += values.double() * active[:, None]
            previous_action = action[:, -6:].clone()
            ended = active & done.bool()
            end_steps[ended] = step + 1
            terminated[ended] = info['terminate'][order][ended].bool()
            active &= ~done.bool()
            if step % 100 == 0 or step == args.steps-1:
                records['trace_time'].append((step+1) * env.control_dt)
                records['trace_survival'].append((~terminated).float().reshape(-1, group_size).mean(1).cpu().numpy())
                records['trace_position'].append([p2[g*group_size:(g+1)*group_size][active[g*group_size:(g+1)*group_size]].mean().sqrt().item()
                                                  for g in range(len(conditions))])
            if step % 1000 == 0 or step == args.steps-1:
                print(json.dumps(dict(condition=args.condition, seed=args.seed, step=step+1,
                                      alive=int(active.sum()), elapsed=round(time.monotonic()-started, 1))), flush=True)
    data = totals.cpu().numpy()
    count = np.maximum(data[:, 0], 1)
    metrics = dict(position_rmse=np.sqrt(data[:, 1]/count), velocity_rmse=np.sqrt(data[:, 2]/count),
                   force_mean=data[:, 3]/count, power_mean=data[:, 4]/count,
                   action_delta_rms=np.sqrt(data[:, 5]/count), srl_return=data[:, 6],
                   activation_mean=data[:, 7]/count, duration=end_steps.cpu().numpy()*env.control_dt,
                   failed=terminated.cpu().numpy(), env_id=np.tile(np.arange(group_size), len(conditions)))
    for group, condition in enumerate(conditions):
        prefix = ('calibration' if args.calibration else condition) + f'_seed{args.seed}'
        sl = slice(group*group_size, (group+1)*group_size)
        local_metrics = {k:v[sl] for k,v in metrics.items()}
        np.savez(args.output / f'{prefix}_metrics.npz', **local_metrics,
             trace_time=records['trace_time'], trace_survival=np.asarray(records['trace_survival'])[:, group],
             trace_position=np.asarray(records['trace_position'])[:, group])
        metadata = dict(condition=condition, seed=args.seed, num_envs=group_size, steps=args.steps,
                    control_dt=env.control_dt, initial_hash=initial_hashes[group], trajectory_hash=trajectory_hashes[group],
                    checkpoint=str(CKPT), checkpoint_epoch=int(checkpoint['epoch']),
                    encoder_alpha=runtime.alpha, deterministic_policy=True,
                    randomized_physical_layout=args.randomize_layout,
                    initial_state_full_batch_commit=True,
                    repeat_first_step_action_max_diff=repeat_first_step_action_max_diff,
                    physical_env_ids=order[sl].cpu().tolist(),
                    elapsed_seconds=time.monotonic()-started,
                    means={k:float(np.mean(v)) for k,v in local_metrics.items() if k != 'env_id'})
        (args.output / f'{prefix}_meta.json').write_text(json.dumps(metadata, indent=2))
        print('COMPLETE ' + json.dumps(metadata), flush=True)
    if args.collect:
        data_prefix = ('calibration' if args.calibration else 'intact') + f'_seed{args.seed}'
        np.savez(args.output / f'{data_prefix}_data.npz',
                 features=np.asarray(records['features']), states=np.asarray(records['states']),
                 active=np.asarray(records['active']), episode=np.asarray(records['episode']),
                 z_all=np.asarray(records['z']), prediction=np.asarray(records['prediction']),
                 obs=np.asarray(records['obs']), sample_steps=np.asarray(records['sample_steps']))
    if args.calibration:
        np.save(args.output / 'calibration_mean.npy', np.concatenate(calibration_z).mean(0))
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
