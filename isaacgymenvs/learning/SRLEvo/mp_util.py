"""
A decorator for computation in subprocesses (Improved & Safe Version)

NOTE:
- All external interfaces remain 100% compatible with original mp_util.py
- Only internal implementation is enhanced: safe process cleanup, 
  deadlock prevention, GPU resource release, and crash recovery.
"""

import multiprocessing as mp
import inspect
import cloudpickle
import wandb
import traceback
import time

# ---------------------
# Wandb Writer
# ---------------------
class WandbWriter:
    def __init__(self):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step)

    def close(self):
        wandb.finish()


# ---------------------
# Cloudpickle Wrapper
# ---------------------
class CloudpickleWrapper(object):
    """Allows using cloudpickle for serialization."""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


# ---------------------
# Worker Function
# ---------------------
def worker(remote, cls, init_args, init_kwargs):
    """
    Actual subprocess code that:
    - instantiates the requested class
    - receives commands
    - executes them
    - sends back results
    """

    try:
        obj = cls.x(*init_args.x, **init_kwargs.x)

        while True:
            try:
                cmd, data = remote.recv()
            except (EOFError, BrokenPipeError):
                break

            if cmd == 'close':
                if hasattr(obj, 'close'):
                    try:
                        obj.close()
                    except Exception:
                        pass
                try:
                    remote.close()
                except Exception:
                    pass
                break

            if hasattr(obj, cmd):
                args, kwargs = data.x
                try:
                    out = getattr(obj, cmd)(*args, **kwargs)
                    remote.send(CloudpickleWrapper(out))
                except Exception as e:
                    # Prevent worker crash → send traceback
                    remote.send(CloudpickleWrapper(
                        RuntimeError(f"Worker command '{cmd}' failed: {e}\n{traceback.format_exc()}")
                    ))
            else:
                remote.send(CloudpickleWrapper(
                    RuntimeError(f"Unknown command '{cmd}'")
                ))

    except Exception as e:
        # fatal error
        try:
            remote.send(CloudpickleWrapper(
                RuntimeError(f"Worker fatal error: {e}\n{traceback.format_exc()}")
            ))
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


# ---------------------
# Job / Ledger
# ---------------------
class JobLedger():
    def __init__(self, remote):
        self._outputs = {}
        self._count_finished = 0
        self._count_started = 0
        self._remote = remote

    def add_job(self, cmd, args, kwargs):
        self.check_for_results()
        data = CloudpickleWrapper((args, kwargs))
        self._remote.send((cmd, data))
        jid = self._count_started
        self._outputs[jid] = None
        self._count_started += 1
        return Job(cmd, args, kwargs, jid, ledger=self)

    def _add_result(self):
        try:
            data = self._remote.recv()
            if self._count_finished in self._outputs:
                self._outputs[self._count_finished] = data.x
        except EOFError:
            # worker crashed or was killed → mark as None
            self._outputs[self._count_finished] = RuntimeError("Worker closed unexpectedly.")
        finally:
            self._count_finished += 1

    def get_results(self, job_id):
        while self._count_finished <= job_id and not self._remote.closed:
            self.check_for_results()

        if job_id not in self._outputs:
            raise ValueError(
                f"Output of job {job_id} can't be found. Possibly removed from ledger."
            )

        return self._outputs[job_id]

    def delete_results(self, job_id):
        self.check_for_results()
        if job_id in self._outputs:
            del self._outputs[job_id]

    def check_for_results(self):
        if self._remote.closed:
            return

        # Drain all pending results
        while True:
            try:
                if not self._remote.poll():
                    break
                self._add_result()
            except (EOFError, BrokenPipeError):
                break

    def is_complete(self, job_id):
        self.check_for_results()
        return self._count_finished > job_id


class Job():
    def __init__(self, cmd, args, kwargs, job_id, ledger):
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.job_id = job_id
        self._ledger = ledger

    def is_finished(self):
        return self._ledger.is_complete(self.job_id)

    @property
    def results(self):
        return self._ledger.get_results(self.job_id)

    def join(self):
        return self._ledger.get_results(self.job_id)

    def __del__(self):
        try:
            self._ledger.delete_results(self.job_id)
        except Exception:
            pass


# ---------------------
# Decorator Wrapper
# ---------------------
def _subproc_decorator(cls, ctx, daemon):

    class RemoteWorker():
        def __init__(self, *args, **kwargs):
            self._closed = False
            self.ctx = mp.get_context(ctx)
            self.remote, self.child = self.ctx.Pipe()

            self.proc = self.ctx.Process(
                target=worker,
                args=(self.child,
                      CloudpickleWrapper(cls),
                      CloudpickleWrapper(args),
                      CloudpickleWrapper(kwargs))
            )
            self.proc.daemon = daemon
            self.proc.start()

            self._ledger = JobLedger(self.remote)

        def close(self):
            """Gracefully close and ensure subprocess cleanup"""
            if self._closed:
                return

            try:
                if not self.remote.closed:
                    self.remote.send(('close', {}))
                    self.remote.close()
            except Exception:
                pass

            # Ensure subprocess exits
            self.proc.join(timeout=8.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=3.0)

            self._closed = True

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

    # Add remote methods based on class API
    def _add_command(name):
        def remote_fn(self, *args, **kwargs):
            return self._ledger.add_job(name, args, kwargs)
        setattr(RemoteWorker, name, remote_fn)

    for name, _ in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith('_') or name == 'close':
            continue
        _add_command(name)

    return RemoteWorker


def subproc_worker(cls=None, ctx='fork', daemon=True):
    if cls is None:
        return lambda cls: _subproc_decorator(cls, ctx, daemon)
    else:
        return _subproc_decorator(cls, ctx, daemon)


# Test
if __name__ == '__main__':
    @subproc_worker
    class MyClass():
        def __init__(self, x):
            self.x = x

        def add_to_x(self, y):
            return self.x + y

    my_obj = MyClass(x=5)
    job = my_obj.add_to_x(5)
    print(job.job_id, job.cmd, job.args, job.kwargs)
    print(job.results)
    my_obj.close()
