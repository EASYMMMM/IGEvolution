#!/usr/bin/env python3
"""Sequential five-stage SRL training, with isolated outputs and verified handoffs.

Stage transitions use hsrl_checkpoint (new phase); interrupted phases use the
top-level checkpoint (resume epoch/optimizer). The trainer stops at epoch > limit,
so an epoch limit of 2000 currently produces a final checkpoint at epoch 2001.
No simulator/training code is patched by this launcher. Python 3.8 compatible.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / 'isaacgymenvs/SRL_Evo_train.py'
ASSET = REPO / 'assets/mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_03dis_mesh.xml'
SOURCE_ASSET = ASSET.with_name('srl_real_hri_v2_HXYK_175_mesh.xml')
HUMANOID = REPO / 'isaacgymenvs/runs/HumanoidAMPMuscleFlat_P002_Finetune_17-12-34-52/nn/HumanoidAMPMuscleFlat_P002_Finetune_17-12-34-55.pth'
TEACHER = REPO / 'isaacgymenvs/runs/SRL_Real_Bot_v2_s4_09-13-35-18/nn/SRL_Real_Bot_v2_s4.pth'
STAGES = ('straight', 'curves', 'omnidirectional', 'HMI', 'HMIEncoder')
DEFAULTS = dict(epochs=[2000, 4000, 4000, 4000, 4000], num_envs=8192,
                device='cuda:0', seed=42, handoff='best', wandb=False,
                humanoid_checkpoint=str(HUMANOID), teacher_checkpoint=str(TEACHER),
                smoke=False)


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def canonical(node):
    return (node.tag, tuple(sorted(node.attrib.items())),
            tuple(canonical(child) for child in node))


def validate_asset():
    """Only model name, root translation and the two local pivots may differ."""
    source = ET.parse(SOURCE_ASSET).getroot()
    target = ET.parse(ASSET).getroot()
    normalized = copy.deepcopy(target)
    normalized.set('model', source.get('model'))
    root = normalized.find('.//body[@name="SRL_root"]')
    if root.get('pos') != '-0.30 0 -0.05':
        raise ValueError('0.3 m asset has an unexpected root position')
    root.set('pos', '-0.60 0 -0.05')
    for name in ('SRL_freejoint_z', 'SRL_freejoint_y'):
        joint = root.find('joint[@name="%s"]' % name)
        if joint.get('pos') != '0.30 0 0':
            raise ValueError('0.3 m asset has an unexpected hinge anchor: ' + name)
        joint.set('pos', '0.60 0 0')
    if canonical(source) != canonical(normalized):
        raise ValueError('0.3 m asset differs from V2 beyond the documented distance/pivot changes')
    meshdir = target.find('compiler').get('meshdir', '')
    for mesh in target.findall('./asset/mesh'):
        path = ASSET.parent / meshdir / mesh.get('file')
        if not path.is_file():
            raise FileNotFoundError('Missing mesh: ' + str(path))
    return sha256(ASSET)


def tensor_shape(state, key, expected):
    if key not in state or tuple(state[key].shape) != expected:
        raise ValueError('Incompatible checkpoint tensor %s; expected %s' % (key, expected))


def load_checkpoint(path, stage=None, bootstrap=None):
    # These files are user-selected local/training checkpoints, not web downloads.
    import torch
    d = torch.load(str(path), map_location='cpu')
    if bootstrap:
        shape = (512, 117) if bootstrap == 'human' else (512, 133)
        out = (58, 256) if bootstrap == 'human' else (6, 128)
        tensor_shape(d['model'], 'a2c_network.actor_mlp.0.weight', shape)
        tensor_shape(d['model'], 'a2c_network.mu.weight', out)
        if bootstrap == 'human':
            tensor_shape(d['amp_input_mean_std'], 'running_mean', (210,))
    else:
        for key in ('model', 'model_srl', 'central_critic', 'optimizer',
                    'optimizer_srl', 'central_critic_opt', 'amp_input_mean_std', 'epoch'):
            if key not in d:
                raise ValueError('Missing joint checkpoint field: ' + key)
        tensor_shape(d['model'], 'a2c_network.actor_mlp.0.weight', (512, 117))
        tensor_shape(d['model_srl'], 'a2c_network.actor_mlp.0.weight', (1024, 198))
        if ('hmi_encoder' in d) != (stage == 5):
            raise ValueError('Encoder mode mismatch for stage %s' % stage)
        if stage == 5:
            for key in ('model', 'optimizer', 'completed_rollouts', 'alpha', 'config'):
                if key not in d['hmi_encoder']:
                    raise ValueError('Incomplete encoder checkpoint: ' + key)
    return dict(epoch=int(d.get('epoch', 0)), encoder=('hmi_encoder' in d),
                bytes=Path(path).stat().st_size)


def hydra_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, Path)):
        return json.dumps(str(value), ensure_ascii=False)
    return str(value)


def stage_dir(run, stage):
    return run / ('stage_%d_%s' % (stage, STAGES[stage - 1]))


def command(settings, run, stage, previous=None, resume=None):
    folder = stage_dir(run, stage)
    prefix = folder / 'checkpoints/policy'
    cfg = {
        'task': 'SRL_Real_HRI', 'headless': True, 'test': False,
        'wandb_activate': settings['wandb'], 'wandb_project': 'SRL_Evo',
        'experiment': 'SRL_V2_03dis_stage%d_%s' % (stage, STAGES[stage - 1]),
        '+full_experiment_name': 'stage%d_%s' % (stage, STAGES[stage - 1]),
        'max_iterations': settings['epochs'][stage - 1], 'seed': settings['seed'],
        'sim_device': settings['device'], 'rl_device': settings['device'],
        'num_envs': settings['num_envs'], 'checkpoint': '',
        'task.env.asset.assetFileName': str(ASSET.relative_to(REPO / 'assets')),
        'task.env.train_stage': (2, 3, 4, 4, 4)[stage - 1],
        'task.env.stateInit': 'Default', 'task.env.episodeLength': 5000,
        'task.env.srl_hmi_control': stage >= 4, 'task.env.HMI_obs_enable': True,
        'task.env.srl_free_actions_num': 5,
        'task.env.srl_max_effort': 150 if stage != 2 else 350,
        'task.env.srl_motor_cost_scale': (0.0, 0.3, 0.5, 0.5, 0.5)[stage - 1],
        'task.env.pelvis_height_reward_scale': 2.0 if stage == 1 else 0.0,
        'task.env.orientation_reward_scale': 3.0 if stage == 1 else 5.0,
        'task.env.no_fly_penalty_scale': 2.0,
        'task.env.gait_similarity_penalty_scale': 2.0 if stage <= 2 else 5.0,
        'task.env.progress_reward_scale': 0.0, 'task.env.vel_tracking_reward_scale': 3.0,
        'task.env.tracking_ang_vel_reward_scale': 2.0,
        'task.env.clearance_penalty_scale': 10.0,
        'task.env.humanoid_share_reward_scale': 2.0 if stage == 1 else 3.0,
        'task.env.contact_force_cost_scale': 0.5 if stage <= 2 else 1.0,
        'train.params.config.train_humanoid': True, 'train.params.config.train_srl': True,
        'train.params.config.train_humanoid_freq': 10,
        'train.params.config.learning_rate': 0.0001,
        'train.params.config.central_critic': True, 'train.params.config.sym_a_loss_coef': 1.0,
        'train.params.config.humanoid_checkpoint': False,
        'train.params.config.srl_teacher_checkpoint': settings['teacher_checkpoint'] if stage == 1 else False,
        'train.params.config.hsrl_checkpoint': False,
        'train.params.config.dagger_loss_coef': 1.0 if stage == 1 else 0.0,
        'train.params.config.dagger_anneal_k': 1e-5 if stage == 1 else 1e-4,
        'train.params.config.hmi_encoder.enabled': stage == 5,
        'train.params.config.hmi_encoder.humanoid_lr_scale': 0.3,
        'train.params.config.hmi_encoder.learning_rate': 0.0003,
        'train.params.config.hmi_encoder.microbatch_size': 2048,
        'train.params.config.train_dir': str(folder / 'rlgames'),
        'train.params.config.model_output_file': str(prefix),
        'train.params.config.save_frequency': 250,
    }
    if resume:
        cfg['checkpoint'] = str(resume)
    elif stage == 1:
        cfg['train.params.config.humanoid_checkpoint'] = settings['humanoid_checkpoint']
    else:
        if not previous:
            raise ValueError('Stage %d requires previous-stage weights' % stage)
        cfg['train.params.config.hsrl_checkpoint'] = str(previous)
    if settings['smoke']:
        # Exercise real loaders, PPO updates, checkpoint writes and all transitions.
        # This is an integration check, not an assessment of walking quality.
        cfg.update({
            # Keep the normal trajectory duration: stage 4's trajectory generator
            # requires an initial 1.5 s stand and cannot fit a 64-step episode.
            'train.params.config.minibatch_size': 512,
            'train.params.config.mini_epochs': 1,
            'train.params.config.amp_obs_demo_buffer_size': 512,
            'train.params.config.amp_replay_buffer_size': 64 * settings['num_envs'],
            'train.params.config.amp_batch_size': 64,
            'train.params.config.amp_minibatch_size': 64,
            'train.params.config.save_frequency': 1,
            'train.params.config.hmi_encoder.microbatch_size': 128,
        })
    return [sys.executable, '-u', str(ENTRY)] + [
        '%s=%s' % (k, hydra_value(v)) for k, v in cfg.items()]


def write_command(path, cmd):
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' +
                    shlex.quote(str(ENTRY.parent)) + '\n' +
                    shlex.join(cmd) + '\n')


def run_child(cmd, log):
    """Tee output and stop the entire training process group on interruption."""
    with log.open('a') as out:
        out.write('\n=== %s ===\n%s\n' % (stamp(), shlex.join(cmd))); out.flush()
        child = subprocess.Popen(cmd, cwd=str(ENTRY.parent), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=True)
        try:
            for line in child.stdout:
                out.write(line); out.flush()
                print(line, end='', flush=True)
            code = child.wait()
            if code:
                raise RuntimeError('Training exited with code %d; inspect %s' % (code, log))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def latest_checkpoint(folder, stage):
    """Resume from greatest saved epoch, not file mtime or best reward."""
    candidates = []
    for p in folder.glob('checkpoints/policy*.pth'):
        if not re.fullmatch(r'policy(?:_best|_\d+)?\.pth', p.name):
            continue
        try:
            info = load_checkpoint(p, stage=stage)
            candidates.append((info['epoch'], p.name == 'policy.pth', str(p)))
        except Exception as e:
            print('Skipping unreadable/incompatible checkpoint %s: %s' % (p, e), flush=True)
    return Path(max(candidates)[2]) if candidates else None


def completed_stage(folder, stage, settings):
    final = folder / 'checkpoints/policy.pth'
    info = load_checkpoint(final, stage=stage)
    if info['epoch'] <= settings['epochs'][stage - 1]:
        raise ValueError('Final checkpoint does not reach this stage epoch limit: ' + str(final))
    best = folder / 'checkpoints/policy_best.pth'
    chosen = final
    if settings['handoff'] == 'best' and best.exists():
        load_checkpoint(best, stage=stage)
        chosen = best
    elif settings['handoff'] == 'best':
        print('No best checkpoint for stage %d; using its validated final checkpoint.' % stage)
    selected = load_checkpoint(chosen, stage=stage)
    return dict(status='complete', finished_at=stamp(), final=str(final), final_epoch=info['epoch'],
                selected=str(chosen), selected_epoch=selected['epoch'], sha256=sha256(chosen))


def execute_pipeline(run, settings, state, child_runner=run_child):
    previous = None
    active = None
    try:
        for stage in range(1, 6):
            key = str(stage); active = None
            folder = stage_dir(run, stage)
            record = state['stages'].get(key, {})
            if record.get('status') == 'complete':
                previous = Path(record['selected'])
                if sha256(previous) != record['sha256']:
                    raise ValueError('Completed-stage checkpoint changed: ' + str(previous))
                load_checkpoint(previous, stage=stage)
                print('Stage %d already complete; verified %s' % (stage, previous), flush=True)
                active = None
                continue
            active = key
            (folder / 'checkpoints').mkdir(parents=True, exist_ok=True)
            final = folder / 'checkpoints/policy.pth'
            # Recover a completed save if interruption happened before state.json commit.
            recovered = False
            if final.exists():
                try:
                    state['stages'][key] = completed_stage(folder, stage, settings)
                    recovered = True
                except Exception as e:
                    print('Final checkpoint not usable as completed stage: %s' % e, flush=True)
            if not recovered:
                resume = latest_checkpoint(folder, stage)
                if record and not resume:
                    print('No usable checkpoint for interrupted stage %d; restarting only this stage.' % stage)
                cmd = command(settings, run, stage, previous, resume)
                state['stages'][key] = dict(status='running', started_at=stamp(),
                                           previous=str(previous) if previous else None,
                                           resumed_from=str(resume) if resume else None)
                save_json(run / 'state.json', state)
                write_command(folder / 'command.sh', cmd)
                # Composition exits before Isaac Gym imports register custom resolvers.
                # Keep interpolations in this audit copy; training resolves them normally.
                with (folder / 'composed_config.yaml').open('w') as f:
                    subprocess.run(cmd + ['--cfg', 'job'], cwd=str(ENTRY.parent),
                                   stdout=f, check=True)
                print('\nStage %d/5 (%s): %s\n' % (stage, STAGES[stage - 1], shlex.join(cmd)), flush=True)
                child_runner(cmd, folder / 'train.log')
                state['stages'][key] = completed_stage(folder, stage, settings)
            previous = Path(state['stages'][key]['selected'])
            save_json(run / 'state.json', state)
            active = None
            print('Stage %d complete. Next-stage input: %s' % (stage, previous), flush=True)
        state['status'] = 'complete'
        save_json(run / 'state.json', state)
        print('\nAll five stages complete. Final selected model: %s' % previous, flush=True)
    except (Exception, KeyboardInterrupt) as e:
        if active:
            state['stages'].setdefault(active, {}).update(status='interrupted_or_failed', error=str(e))
        state['status'] = 'interrupted_or_failed'
        save_json(run / 'state.json', state)
        raise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path, help='独立输出目录；续训时指定同一目录')
    p.add_argument('--resume', action='store_true', help='跳过完成阶段，从未完成阶段最新有效 checkpoint 续训')
    p.add_argument('--dry-run', action='store_true', help='验证 XML/前置模型并打印全部命令，不启动训练')
    p.add_argument('--smoke', action='store_true', default=None, help='32 环境、每阶段 max_iterations=1 的真实联通检查')
    p.add_argument('--epochs', nargs=5, type=int, metavar=('S1', 'S2', 'S3', 'S4', 'S5'))
    p.add_argument('--num-envs', type=int, help='默认 8192；普通模式须使 32×环境数可被 minibatch 65536 整除')
    p.add_argument('--device', help='默认 cuda:0')
    p.add_argument('--seed', type=int, help='默认 42')
    p.add_argument('--handoff', choices=['best', 'last'], help='跨阶段默认取 best，无 best 时取 final')
    p.add_argument('--wandb', action='store_true', default=None, help='启用 W&B；默认只保存本地 TensorBoard/日志')
    p.add_argument('--humanoid-checkpoint', type=Path)
    p.add_argument('--teacher-checkpoint', type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    supplied = {k: str(v.resolve()) if isinstance(v, Path) else v
                for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    if args.resume and not args.run_dir:
        raise ValueError('--resume requires --run-dir')
    run = (args.run_dir or REPO / 'run_data/srl_v2_03dis' /
           (datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])).resolve()
    if args.resume:
        state = json.loads((run / 'state.json').read_text())
        settings = state['settings']
        if state.get('version') != 1:
            raise ValueError('Unknown pipeline state version')
        if any(settings[k] != v for k, v in supplied.items()):
            raise ValueError('Resume settings differ from saved settings; use a new run directory for a new experiment')
    else:
        settings = dict(DEFAULTS, **supplied)
        if settings['smoke']:
            settings.setdefault('epochs', [1] * 5)
            if 'epochs' not in supplied: settings['epochs'] = [1] * 5
            if 'num_envs' not in supplied: settings['num_envs'] = 32
        state = dict(version=1, created_at=stamp(), settings=settings, stages={}, status='pending')
        if run.exists() and any(run.iterdir()):
            raise ValueError('Run directory is not empty; use --resume or choose a new directory: ' + str(run))
    if min(settings['epochs']) < 1 or settings['num_envs'] < 1 or settings['seed'] < 0:
        raise ValueError('Use positive epochs/environment count and a nonnegative reproducible seed')
    mb = 512 if settings['smoke'] else 65536
    if (32 * settings['num_envs']) % mb:
        raise ValueError('horizon_length(32) × num_envs must be divisible by minibatch_size(%d)' % mb)
    fingerprints = {'asset': validate_asset()}
    for name, kind in (('humanoid_checkpoint', 'human'), ('teacher_checkpoint', 'teacher')):
        path = Path(settings[name])
        load_checkpoint(path, bootstrap=kind)
        fingerprints[name] = sha256(path)
    if args.resume and state['fingerprints'] != fingerprints:
        raise ValueError('Asset or bootstrap checkpoint content changed since this run started')
    state['fingerprints'] = fingerprints
    print('Run directory: %s\nMode: %s\nCheckpoint handoff: %s' %
          (run, 'SMOKE CHECK ONLY' if settings['smoke'] else 'FULL TRAINING', settings['handoff']), flush=True)
    if args.dry_run:
        previous = None
        for stage in range(1, 6):
            print('\n# Stage %d/5: %s' % (stage, STAGES[stage - 1]))
            print(shlex.join(command(settings, run, stage, previous)))
            # Actual execution resolves best/last only after successful stage completion.
            previous = stage_dir(run, stage) / 'checkpoints' / (
                'policy_best.pth' if settings['handoff'] == 'best' else 'policy.pth')
        print('\nDry run passed. Commands are a plan; best falls back to final if unavailable.')
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reload under the lock so two simultaneous resume requests cannot use stale state.
        if args.resume:
            state = json.loads((run / 'state.json').read_text())
        elif (run / 'state.json').exists():
            raise ValueError('Another launcher has already initialized this run')
        state['status'] = 'running'
        save_json(run / 'state.json', state)
        (run / ASSET.name).write_bytes(ASSET.read_bytes())  # audit snapshot; meshes remain in assets/
        execute_pipeline(run, settings, state)


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt('Received signal %s' % signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('\nPipeline stopped: %s\nResume with --resume --run-dir <same directory>.' % exc, file=sys.stderr)
        sys.exit(130 if isinstance(exc, KeyboardInterrupt) else 1)
