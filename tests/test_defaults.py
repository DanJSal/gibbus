"""Cross-cutting default, debug, and fallback-ledger contracts."""

import threading

import pytest

import gibbus._defaults as defaults


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_env_flag_false_spellings(monkeypatch, value):
    """Common false spellings do not enable debug-style environment flags."""
    monkeypatch.setenv("GIBBUS_TEST_FLAG", value)
    assert defaults._env_flag("GIBBUS_TEST_FLAG") is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_flag_true_spellings(monkeypatch, value):
    """Explicit true spellings enable debug-style environment flags."""
    monkeypatch.setenv("GIBBUS_TEST_FLAG", value)
    assert defaults._env_flag("GIBBUS_TEST_FLAG") is True


def test_suppressed_failure_ledger_is_thread_local(monkeypatch):
    """A worker thread cannot clear or append to the caller's ledger."""
    monkeypatch.setattr(defaults, "DEBUG", False)
    monkeypatch.setattr(defaults, "DEBUG_STRICT", False)
    defaults.clear_suppressed_failures()
    defaults._reraise_if_debug(RuntimeError("main"), "main", routine=True)

    worker_records = []

    def worker():
        defaults.clear_suppressed_failures()
        defaults._reraise_if_debug(
            RuntimeError("worker"), "worker", routine=True)
        worker_records.extend(defaults.suppressed_failures())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert [r["context"] for r in worker_records] == ["worker"]
    assert [r["context"] for r in defaults.suppressed_failures()] == ["main"]
    defaults.clear_suppressed_failures()
