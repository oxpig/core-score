"""Execute the installed OST CLI, reusing its interpreter for ligand comparisons.

No coordinates, parsed structures, or scores are cached. Each process/thread owns its
worker. Unsupported OST installations fall back to a fresh CLI process in auto mode.
"""

import atexit
from contextlib import contextmanager
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid

_CLIENTS = {}
_UNAVAILABLE = {}
_AUDIT_ENV = "_CORE_SCORE_OST_AUDIT_DIR"


def mode():
    value = os.environ.get("CORE_SCORE_OST_MODE", "auto")
    if value not in ("auto", "persistent", "subprocess"):
        raise ValueError("CORE_SCORE_OST_MODE must be auto, persistent, or subprocess")
    return value


def environment():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def stop(process):
    """Kill the wrapper and its children, including after a timeout or interruption."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def audit(event):
    directory = os.environ.get(_AUDIT_ENV)
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{uuid.uuid4().hex}.json").write_text(json.dumps(event, indent=2) + "\n")


@contextmanager
def session(details):
    """Keep execution diagnostics and close workers even when a pipeline stage fails."""
    mode()  # Validate before doing any work.
    previous = os.environ.get(_AUDIT_ENV)
    os.environ[_AUDIT_ENV] = str(Path(details) / "ost_execution")
    try:
        yield
    finally:
        close_clients()
        if previous is None:
            os.environ.pop(_AUDIT_ENV, None)
        else:
            os.environ[_AUDIT_ENV] = previous


class Unavailable(RuntimeError):
    pass


class Client:
    def __init__(self, timeout):
        self.owner = os.getpid()
        self.connection, child = multiprocessing.Pipe()
        self.process = None
        worker = Path(__file__).with_name("ost_worker.py")
        try:
            # Launch through OST itself: it selects its own Python and chemical library.
            # The worker uses only stdlib + OST, so CORE need not be installed in that Python.
            self.process = subprocess.Popen(
                ["ost", str(worker), str(child.fileno())], env=environment(),
                pass_fds=(child.fileno(),), start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        try:
            if not self.connection.poll(min(timeout, 30) if timeout is not None else 30):
                raise Unavailable("OST worker did not start within the startup time limit")
            ready = self.connection.recv()
            if not ready.get("ready"):
                raise Unavailable(ready.get("error", "OST worker could not load the CLI"))
        except (EOFError, OSError) as exc:
            self.close()
            raise Unavailable(f"OST worker startup failed: {exc}") from exc
        except BaseException:
            self.close()
            raise
        # multiprocessing workers do not run normal atexit callbacks.
        self.finalizer = Finalize(self, self.close, exitpriority=10)

    def close(self):
        if self.owner != os.getpid():
            return
        self.connection.close()
        if self.process is not None:
            stop(self.process)
            self.process = None

    def call(self, arguments, timeout):
        self.connection.send(arguments)
        if not self.connection.poll(timeout):
            raise subprocess.TimeoutExpired(["ost", "compare-ligand-structures", *arguments], timeout)
        return self.connection.recv()


def ordinary(command, timeout):
    process = subprocess.Popen(command, env=environment(), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        stop(process)
        process.communicate()
        raise
    return dict(command=command, returncode=process.returncode, stdout=stdout,
                stderr=stderr, transport="subprocess")


def run(arguments, output, timeout):
    command = ["ost", *map(str, arguments), "-o", str(output)]
    selected = mode()
    if arguments[0] != "compare-ligand-structures" or selected == "subprocess":
        return ordinary(command, timeout)
    owner = (os.getpid(), threading.get_ident())
    if os.name != "posix":
        _UNAVAILABLE[owner] = "Persistent OST requires POSIX file descriptor passing"
    if owner not in _UNAVAILABLE and owner not in _CLIENTS:
        try:
            _CLIENTS[owner] = Client(timeout)
        except Unavailable as exc:
            _UNAVAILABLE[owner] = str(exc)
            audit(dict(event="persistent_unavailable", reason=str(exc), mode=selected))
    if owner in _UNAVAILABLE:
        if selected == "persistent":
            raise Unavailable(_UNAVAILABLE[owner])
        return dict(ordinary(command, timeout), persistent_unavailable=_UNAVAILABLE[owner])
    client = _CLIENTS[owner]
    try:
        code, stdout, stderr = client.call(command[2:], timeout)
    except BaseException:
        client.close()
        _CLIENTS.pop(owner, None)
        raise
    return dict(command=command, returncode=code, stdout=stdout, stderr=stderr,
                transport="persistent_original_cli", worker_pid=client.process.pid)


def close_clients():
    for client in list(_CLIENTS.values()):
        client.close()
    _CLIENTS.clear()
    _UNAVAILABLE.clear()


def after_fork():
    # Children must not keep their parent's RPC endpoints alive or kill its workers.
    for client in _CLIENTS.values():
        client.connection.close()
    _CLIENTS.clear()
    _UNAVAILABLE.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=after_fork)
atexit.register(close_clients)
