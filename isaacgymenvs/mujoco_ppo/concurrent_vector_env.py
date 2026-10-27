"""Ordered, synchronous MuJoCo sampling with optional persistent CPU workers."""

import copy
import multiprocessing as mp
import os
import traceback
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np


def _make_env(config, history_len):
    from mujoco_ppo.srl_mujoco_concurrent_env import SRLMujocoConcurrentEnv
    return SRLMujocoConcurrentEnv(config, estimator_history_len=history_len)


def _metadata(env):
    return {
        "cfg": copy.deepcopy(env.cfg), "act_dim": env.act_dim,
        "motor_damping": env.model.dof_damping[env.srl_dof_indices].copy(),
    }


def _worker(connection, configurations, history_len):
    try:
        envs = [_make_env(copy.deepcopy(config), history_len) for config in configurations]
        connection.send((True, [_metadata(env) for env in envs]))
        while True:
            command, values = connection.recv()
            if command == "close":
                break
            if command == "step":
                result = [env.step(action) for env, action in zip(envs, values)]
            elif command == "reset":
                result = [(index, envs[index].reset(seed=seed)) for index, seed in values]
            elif command == "progress":
                for env in envs:
                    env.set_dr_progress(values)
                result = None
            elif command == "configure":
                for env, changes in zip(envs, values):
                    for name, value in changes.items():
                        setattr(env.cfg, name, value)
                result = None
            else:
                raise ValueError("Unknown environment command: {}".format(command))
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ConcurrentVectorEnv:
    """Never auto-resets: terminal observations must reach the PPO buffer first."""

    def __init__(self, configurations, history_len=10, workers=0,
                 worker_threads=1, timeout=120.0):
        if (isinstance(workers, bool) or int(workers) != workers or workers < 0
                or int(worker_threads) != worker_threads or worker_threads < 1
                or not np.isfinite(timeout) or timeout <= 0 or not configurations):
            raise ValueError("Invalid environment worker configuration")
        self.workers = min(int(workers), len(configurations))
        self.timeout = float(timeout)
        self.connections, self.processes, self.groups = [], [], []
        self.envs, self.metadata = [], []
        self.closed = False
        try:
            if not self.workers:
                self.envs = [_make_env(copy.deepcopy(c), history_len) for c in configurations]
                self.metadata = [_metadata(env) for env in self.envs]
            else:
                context = mp.get_context("spawn")
                # Spawn inherits these limits before NumPy/BLAS imports in children.
                names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                previous = {name: os.environ.get(name) for name in names}
                try:
                    for name in names:
                        os.environ[name] = str(worker_threads)
                    for group in np.array_split(np.arange(len(configurations)), self.workers):
                        ids = group.tolist()
                        parent, child = context.Pipe()
                        process = context.Process(
                            target=_worker,
                            args=(child, [configurations[i] for i in ids], history_len),
                            daemon=True,
                        )
                        self.connections.append(parent)
                        self.processes.append(process)
                        self.groups.append(ids)
                        try:
                            process.start()
                        finally:
                            child.close()
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                self.metadata = [item for conn in self.connections for item in self._receive(conn)]
                self.envs = [SimpleNamespace(cfg=item["cfg"], act_dim=item["act_dim"])
                             for item in self.metadata]
            self._sent_configs = [asdict(env.cfg) for env in self.envs]
        except BaseException:
            self.close()
            raise

    def _receive(self, connection):
        if not connection.poll(self.timeout):
            raise TimeoutError("MuJoCo worker timed out after {} s".format(self.timeout))
        try:
            success, value = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("MuJoCo worker exited unexpectedly") from error
        if not success:
            raise RuntimeError("MuJoCo worker failed:\n" + value)
        return value

    def _broadcast(self, command, values):
        for conn, value in zip(self.connections, values):
            conn.send((command, value))
        return [self._receive(conn) for conn in self.connections]

    def set_dr_progress(self, progress):
        if self.workers:
            self._broadcast("progress", [float(progress)] * self.workers)
        else:
            for env in self.envs:
                env.set_dr_progress(float(progress))

    def sync_configs(self):
        if not self.workers:
            return
        changes = []
        for index, env in enumerate(self.envs):
            current = asdict(env.cfg)
            changes.append({key: value for key, value in current.items()
                            if value != self._sent_configs[index][key]})
            self._sent_configs[index] = current
        if any(changes):
            self._broadcast("configure", [[changes[i] for i in ids] for ids in self.groups])

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (len(self.envs), self.envs[0].act_dim):
            raise ValueError("Actions must contain one action per environment")
        if not self.workers:
            return [env.step(action) for env, action in zip(self.envs, actions)]
        return [item for group in self._broadcast(
            "step", [actions[ids] for ids in self.groups]) for item in group]

    def reset(self, indices=None, seeds=None):
        ids = list(range(len(self.envs))) if indices is None else list(indices)
        if len(ids) != len(set(ids)) or any(i < 0 or i >= len(self.envs) for i in ids):
            raise ValueError("Invalid reset environment indices")
        seeds = [None] * len(ids) if seeds is None else list(seeds)
        if len(seeds) != len(ids):
            raise ValueError("One reset seed is required per environment")
        if not ids:
            return {}
        if not self.workers:
            return {i: self.envs[i].reset(seed=seed) for i, seed in zip(ids, seeds)}
        requested = dict(zip(ids, seeds))
        requests = [[(local, requested[global_id]) for local, global_id in enumerate(group)
                     if global_id in requested] for group in self.groups]
        replies = self._broadcast("reset", requests)
        return {group[local]: value for group, reply in zip(self.groups, replies)
                for local, value in reply}

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self.connections:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=3)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
        for conn in self.connections:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
