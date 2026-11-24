"""Failure/restart and handoff checks without requiring Isaac Gym or a GPU."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'pipeline', ROOT / 'tools/srl_training/train_five_stages.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def options(cmd):
    return {key: value for key, value in (arg.split('=', 1) for arg in cmd[3:])}


def fake_checkpoint(path, **kwargs):
    return json.loads(Path(path).read_text())


class PipelineTests(unittest.TestCase):
    def test_asset_changes_preserve_v2_and_humanoid_side_pivots(self):
        pipeline.validate_asset()  # Also checks every referenced mesh exists.
        source = ET.parse(pipeline.SOURCE_ASSET).getroot()
        target = ET.parse(pipeline.ASSET).getroot()
        for name in ('SRL_freejoint_y', 'SRL_freejoint_z'):
            pivots = []
            for model in (source, target):
                root = model.find('.//body[@name="SRL_root"]')
                joint = root.find('joint[@name="%s"]' % name)
                pivots.append(tuple(a + b for a, b in zip(
                    map(float, root.get('pos').split()),
                    map(float, joint.get('pos').split()))))
            self.assertEqual(pivots[0], pivots[1])

    def test_interruption_resumes_only_failed_phase_and_preserves_handoffs(self):
        settings = copy.deepcopy(pipeline.DEFAULTS)
        settings['epochs'] = [1] * 5
        state = {'stages': {}, 'status': 'running'}
        calls = []
        fail_stage_two = [True]
        with tempfile.TemporaryDirectory(prefix='SRL path with spaces ') as tmp:
            run = Path(tmp)

            def child(cmd, log):
                cfg = options(cmd)
                stage = next(i for i in range(1, 6)
                             if log.parent == pipeline.stage_dir(run, i))
                calls.append((stage, cfg))
                prefix = Path(json.loads(cfg['train.params.config.model_output_file']))
                if stage == 2 and fail_stage_two[0]:
                    fail_stage_two[0] = False
                    prefix.with_name('policy_1.pth').write_text('{"epoch": 1}')
                    raise RuntimeError('simulated process interruption')
                prefix.with_suffix('.pth').write_text('{"epoch": 2}')
                # Stage 1 intentionally has no best, exercising final fallback.
                if stage != 1:
                    prefix.with_name('policy_best.pth').write_text('{"epoch": 1}')

            with patch.object(pipeline, 'load_checkpoint', side_effect=fake_checkpoint), \
                    patch.object(pipeline.subprocess, 'run'), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    pipeline.execute_pipeline(run, settings, state, child)
                self.assertEqual([s for s, _ in calls], [1, 2])
                self.assertEqual(state['stages']['1']['status'], 'complete')
                self.assertEqual(state['stages']['2']['status'], 'interrupted_or_failed')
                pipeline.execute_pipeline(run, settings, state, child)
                self.assertEqual([s for s, _ in calls], [1, 2, 2, 3, 4, 5])
                self.assertEqual(state['status'], 'complete')
                # A completed run is idempotent: no training is repeated.
                pipeline.execute_pipeline(run, settings, state, child)
                self.assertEqual(len(calls), 6)
                selected = Path(state['stages']['1']['selected'])
                selected.write_text('{"epoch": 99}')
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                        pipeline.execute_pipeline(run, settings, state, child)
                    # A checksum failure must never turn a completed phase into
                    # a restartable phase and bypass the guard on the next try.
                    self.assertEqual(state['stages']['1']['status'], 'complete')
                    self.assertEqual(len(calls), 6)

            first, entering_two, resuming_two, three, four, five = [c for _, c in calls]
            self.assertEqual(json.loads(entering_two['train.params.config.hsrl_checkpoint']),
                             state['stages']['1']['selected'])
            self.assertTrue(state['stages']['1']['selected'].endswith('/policy.pth'))
            self.assertEqual(entering_two['checkpoint'], '""')
            self.assertTrue(json.loads(resuming_two['checkpoint']).endswith('/policy_1.pth'))
            self.assertEqual(resuming_two['train.params.config.hsrl_checkpoint'], 'False')
            self.assertEqual(json.loads(three['train.params.config.hsrl_checkpoint']),
                             state['stages']['2']['selected'])
            self.assertTrue(state['stages']['2']['selected'].endswith('/policy_best.pth'))
            for index, cfg in enumerate((first, entering_two, three, four, five), 1):
                self.assertEqual(cfg['task.env.train_stage'], str((2, 3, 4, 4, 4)[index - 1]))
                self.assertEqual(cfg['task.env.srl_hmi_control'], str(index >= 4))
                self.assertEqual(cfg['train.params.config.hmi_encoder.enabled'], str(index == 5))
                if index > 1:
                    self.assertEqual(cfg['train.params.config.humanoid_checkpoint'], 'False')
                    self.assertEqual(cfg['train.params.config.srl_teacher_checkpoint'], 'False')

    def test_latest_epoch_not_best_filename_and_incomplete_final_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ckpts = folder / 'checkpoints'
            ckpts.mkdir()
            (ckpts / 'policy_250.pth').write_text('{"epoch": 250}')
            (ckpts / 'policy_best.pth').write_text('{"epoch": 120}')
            (ckpts / 'policy_500.pth').write_text('interrupted write')
            (ckpts / 'policy_other_run.pth').write_text('{"epoch": 9000}')
            with patch.object(pipeline, 'load_checkpoint', side_effect=fake_checkpoint), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.latest_checkpoint(folder, 1).name, 'policy_250.pth')
                (ckpts / 'policy.pth').write_text('{"epoch": 250}')
                with self.assertRaisesRegex(ValueError, 'epoch limit'):
                    pipeline.completed_stage(folder, 1, pipeline.DEFAULTS)


if __name__ == '__main__':
    unittest.main()
