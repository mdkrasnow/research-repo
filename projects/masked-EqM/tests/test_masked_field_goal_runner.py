import importlib.util
from pathlib import Path


def _load_goal_runner():
    path = Path(__file__).resolve().parents[1] / "experiments" / "masked_eqm_field_shaping" / "goal_runner.py"
    spec = importlib.util.spec_from_file_location("masked_field_goal_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pending_job_without_log_returns_empty_tail(monkeypatch):
    goal_runner = _load_goal_runner()
    commands = []

    def fake_remote(command):
        commands.append(command)
        if not command.endswith("2>/dev/null || true"):
            raise RuntimeError("a missing pending-job log would fail the monitor")
        return ""

    monkeypatch.setattr(goal_runner, "remote", fake_remote)
    result = goal_runner.tail(
        {"log_pattern": "/tmp/masked-field-eval-{job_id}.out"},
        "37247258",
    )

    assert result == ""
    assert commands == ["tail -20 /tmp/masked-field-eval-37247258.out 2>/dev/null || true"]
