"""Worker launched by `ost`, executing that installation's original ligand CLI.

Keep this file independent of the rest of core_score: OST may use a separate Python.
"""

import contextlib
import io
from multiprocessing.connection import Connection
import os
from pathlib import Path
import runpy
import sys
import traceback


def serve(connection, ost, script, api):
    while True:
        try:
            arguments = connection.recv()
        except EOFError:
            break
        if arguments is None:
            break
        stdout, stderr = io.StringIO(), io.StringIO()
        original_argv = sys.argv
        had_traceback = hasattr(sys, "tracebacklimit")
        original_traceback = getattr(sys, "tracebacklimit", None)
        push_sink, pop_sink = ost.PushLogSink, ost.PopLogSink
        push_level, pop_level = ost.PushVerbosityLevel, ost.PopVerbosityLevel
        depths = [0, 0]

        def sink(value):
            push_sink(value)
            depths[0] += 1

        def unsink():
            pop_sink()
            depths[0] -= 1

        def level(value):
            push_level(value)
            depths[1] += 1

        def unlevel():
            pop_level()
            depths[1] -= 1

        ost.PushLogSink, ost.PopLogSink = sink, unsink
        ost.PushVerbosityLevel, ost.PopVerbosityLevel = level, unlevel
        code = 0
        try:
            sys.argv = [str(script), *arguments]
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                api["_Main"]()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if code:
                print(str(exc), file=stderr)
        except BaseException:
            code = 1
            traceback.print_exc(file=stderr)
        finally:
            for _ in range(max(0, depths[0])):
                pop_sink()
            for _ in range(max(0, depths[1])):
                pop_level()
            ost.PushLogSink, ost.PopLogSink = push_sink, pop_sink
            ost.PushVerbosityLevel, ost.PopVerbosityLevel = push_level, pop_level
            sys.argv = original_argv
            if had_traceback:
                sys.tracebacklimit = original_traceback
            elif hasattr(sys, "tracebacklimit"):
                del sys.tracebacklimit
        connection.send((code, stdout.getvalue(), stderr.getvalue()))


def main():
    connection = Connection(int(sys.argv[1]))
    try:
        import ost
        root = Path(os.environ.get("OST_EXEC_DIR", Path(sys.prefix) / "libexec/openstructure"))
        script = root / "ost-compare-ligand-structures"
        api = runpy.run_path(str(script), init_globals={
            name: getattr(ost, name) for name in ("LogScript", "LogInfo", "LogWarning", "LogError")})
        if not callable(api.get("_Main")):
            raise RuntimeError("Installed OST ligand CLI has no callable _Main")
    except BaseException:
        connection.send(dict(ready=False, error=traceback.format_exc()))
        connection.close()
        return
    connection.send(dict(ready=True))
    try:
        serve(connection, ost, script, api)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
