"""Transport lifecycle tests use a tiny fake OST executable, not a scoring mock."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from core_score.analysis import ost, transport


@pytest.fixture
def fake_ost(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("Persistent workers require POSIX")
    program = tmp_path / "ost"
    program.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
from pathlib import Path
from multiprocessing.connection import Connection

def score(args):
    if '--sleep' in args:
        time.sleep(10)
    if '--brief-sleep' in args:
        time.sleep(0.3)
    if '--crash' in args:
        os._exit(3)
    out = Path(args[args.index('-o') + 1])
    source = Path(args[args.index('-m') + 1])
    out.write_text(json.dumps({'status': 'SUCCESS', 'value': source.read_text()}))

if sys.argv[1] == 'compare-ligand-structures':
    score(sys.argv[2:])
else:
    connection = Connection(int(sys.argv[2]))
    connection.send({'ready': not Path(__file__).with_name('unavailable').exists(),
                     'error': 'No callable CLI'})
    while True:
        try:
            args = connection.recv()
        except EOFError:
            break
        score(args)
        connection.send((0, '', ''))
''')
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("CORE_SCORE_OST_MODE", "persistent")
    source = tmp_path / "model.cif"
    source.write_text("first coordinates")
    transport.close_clients()
    yield ["compare-ligand-structures", "-m", str(source)]
    transport.close_clients()


def test_reuses_interpreter_but_reads_changed_structure_each_time(fake_ost):
    first, a = ost.execute(fake_ost, 5)
    Path(fake_ost[-1]).write_text("changed coordinates")
    second, b = ost.execute(fake_ost, 5)
    assert a["worker_pid"] == b["worker_pid"]
    assert (first["value"], second["value"]) == ("first coordinates", "changed coordinates")
    pid = a["worker_pid"]
    transport.close_clients()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_timeout_replaces_worker_and_records_failure(fake_ost, tmp_path):
    _, first = ost.execute(fake_ost, 5)
    with transport.session(tmp_path):
        with pytest.raises(ost.OSTError, match="TimeoutExpired") as failure:
            ost.execute([*fake_ost, "--sleep"], 0.2)
        assert len(failure.value.evidence["timeout_retries"]) == 1
        records = [json.loads(p.read_text()) for p in (tmp_path / "ost_execution").glob("*.json")]
        assert [r["retry"] for r in sorted(records, key=lambda r: r["attempt"])] == [False]
        _, fresh = ost.execute(fake_ost, 5)
        assert fresh["worker_pid"] != first["worker_pid"]
    assert not transport._CLIENTS


def test_worker_crash_is_not_a_score_and_next_call_recovers(fake_ost):
    with pytest.raises((EOFError, ConnectionError)):
        ost.execute([*fake_ost, "--crash"], 5)
    result, _ = ost.execute(fake_ost, 5)
    assert result["status"] == "SUCCESS"


def test_auto_fallback_is_explicit_and_strict_mode_does_not_fallback(fake_ost, tmp_path, monkeypatch):
    (tmp_path / "unavailable").touch()
    monkeypatch.setenv("CORE_SCORE_OST_MODE", "auto")
    with transport.session(tmp_path):
        result, execution = ost.execute(fake_ost, 5)
        assert result["status"] == "SUCCESS"
        assert execution["transport"] == "subprocess"
        assert execution["persistent_unavailable"] == "No callable CLI"
        assert len(list((tmp_path / "ost_execution").glob("*.json"))) == 1
        monkeypatch.setenv("CORE_SCORE_OST_MODE", "persistent")
        with pytest.raises(transport.Unavailable):
            ost.execute(fake_ost, 5)


def test_explicit_timeout_does_not_retry_or_use_partial_output(tmp_path):
    calls = []

    def run(arguments, output, timeout):
        calls.append((arguments, output, timeout))
        if len(calls) == 1:
            output.write_text('{"stale": true}')
            raise subprocess.TimeoutExpired("ost", timeout)
        output.write_text('{"status": "SUCCESS"}')
        return dict(returncode=0, stdout="", stderr="")

    with patch.object(transport, "run", run):
        with pytest.raises(ost.OSTError, match="TimeoutExpired"):
            ost.execute(["compare-ligand-structures", "-m", "pose.cif"], 2)
        assert len(calls) == 1
        result, evidence = ost.execute(["compare-ligand-structures", "-m", "pose.cif"], None)
    assert result == {"status": "SUCCESS"}
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]
    assert evidence["timeout_retries"] == []


@pytest.mark.parametrize("mode", ["persistent", "subprocess"])
def test_default_no_deadline_allows_comparison_to_complete(fake_ost, monkeypatch, mode):
    monkeypatch.setenv("CORE_SCORE_OST_MODE", mode)
    result, evidence = ost.execute([*fake_ost, "--brief-sleep"])
    assert result["value"] == "first coordinates"
    assert evidence["timeout_retries"] == []


def test_session_cleanup_on_exception(fake_ost, tmp_path):
    with pytest.raises(ValueError):
        with transport.session(tmp_path):
            _, execution = ost.execute(fake_ost, 5)
            raise ValueError("scoring stage failed")
    with pytest.raises(ProcessLookupError):
        os.kill(execution["worker_pid"], 0)


def test_no_retry_for_invalid_json_or_non_timeout_failure():
    def corrupt(arguments, output, timeout):
        output.write_text("not JSON")
        return dict(returncode=1, stdout="", stderr="invalid")

    with patch.object(transport, "run", side_effect=corrupt) as call:
        with pytest.raises(ost.OSTError, match="Invalid OST output"):
            ost.execute(["compare-ligand-structures"], 1)
    assert call.call_count == 1
