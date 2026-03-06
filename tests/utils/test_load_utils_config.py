"""Tests for environment-based global config resolution."""

from __future__ import annotations

import pytest

from app.utils.load_utils import load_global_config, resolve_global_config_path


def test_resolve_global_config_path_defaults_to_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to app/global.yaml when REPORT_ASSISTANT_ENV is unset."""

    monkeypatch.delenv("REPORT_ASSISTANT_ENV", raising=False)
    assert resolve_global_config_path().as_posix().endswith("app/global.yaml")


def test_load_global_config_uses_production_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load app/global.prod.yaml when REPORT_ASSISTANT_ENV is production."""

    monkeypatch.setenv("REPORT_ASSISTANT_ENV", "production")
    config = load_global_config()
    assert "postgres:5432" in str(config.POSTGRESQL_URL)


def test_resolve_global_config_path_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported REPORT_ASSISTANT_ENV values with a clear error."""

    monkeypatch.setenv("REPORT_ASSISTANT_ENV", "staging")
    with pytest.raises(ValueError, match="REPORT_ASSISTANT_ENV must be one of"):
        resolve_global_config_path()
