import multiprocessing
import os
import signal
import unittest
from unittest.mock import patch

from isaacgymenvs.learning.SRLEvo import designer_opt
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker


class MorphologySearchTests(unittest.TestCase):
    def test_fixed_values_and_bounds_apply_to_the_baseline_and_all_candidates(self):
        baseline = {"length": 1.0, "hinge": 1, "count": 0}
        space = {"length": ("real", 0.2, 0.6), "hinge": ("fixed", 0),
                 "count": ("int", 2.0, 5.0)}
        for optimizer, options in (
            (designer_opt.RandomOptimizer, {}),
            (designer_opt.GeneticAlgorithmOptimizer, {"population_size": 3}),
            (designer_opt.GeneticAlgorithmOptimizer_v2, {"population_size": 3}),
        ):
            with self.subTest(optimizer=optimizer.__name__):
                candidates = []

                def evaluate(candidate, **kwargs):
                    candidates.append(candidate.copy())
                    return candidate["length"]

                search = optimizer(baseline, space, evaluate, num_iterations=2,
                                   wandb_log=None, **options)
                with patch.object(designer_opt.wandb, "log", side_effect=AssertionError):
                    history = search.optimize()
                self.assertEqual(len(history), 2)
                self.assertEqual(search.base_params, {"length": 0.6, "hinge": 0, "count": 2})
                for candidate in candidates:
                    self.assertLessEqual(0.2, candidate["length"])
                    self.assertLessEqual(candidate["length"], 0.6)
                    self.assertEqual(candidate["hinge"], 0)
                    self.assertIsInstance(candidate["count"], int)
                    self.assertIn(candidate["count"], range(2, 6))
                self.assertEqual(baseline, {"length": 1.0, "hinge": 1, "count": 0})

    def test_bo_passes_the_configured_fixed_design_to_the_objective(self):
        candidates = []
        logs = []

        def evaluate(candidate, **kwargs):
            candidates.append(candidate.copy())
            return 5.0

        optimizer = designer_opt.BayesianOptimizer(
            {"offset": 0.6, "hinge": 1}, {"offset": ("real", 0.3, 0.3), "hinge": ("fixed", 0)},
            evaluate, num_iterations=2, wandb_log=logs.append)
        optimizer.optimize()
        self.assertEqual(candidates, [{"offset": 0.3, "hinge": 0}] * 2)
        self.assertEqual(logs[-1]["BO/best_reward"], 5.0)


class WorkerOperations:
    def square(self, x):
        return x * x

    def fail(self):
        raise ValueError("candidate failed")

    def crash(self):
        os._exit(3)


class WorkerTests(unittest.TestCase):
    def test_join_returns_results_in_submission_order_and_preserves_errors(self):
        runner = subproc_worker(WorkerOperations, ctx="spawn")()
        try:
            first = runner.square(2)
            failed = runner.fail()
            last = runner.square(3)
            self.assertEqual(last.join(), 9)
            self.assertEqual(first.results, 4)
            self.assertIsInstance(failed.results, RuntimeError)
            self.assertIn("candidate failed", str(failed.join()))
        finally:
            runner.close()

    def test_native_worker_exit_wakes_join(self):
        def timeout(signum, frame):
            raise TimeoutError("worker EOF did not wake join()")

        runner = subproc_worker(WorkerOperations, ctx="spawn")()
        handler = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(20)
            job = runner.crash()
            self.assertIsInstance(job.join(), RuntimeError)
            self.assertIn("Worker closed", str(job.results))
            self.assertTrue(job.is_finished())
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)
            runner.close()
        self.assertEqual(runner.proc.exitcode, 3)
        self.assertNotIn(runner.proc, multiprocessing.active_children())
