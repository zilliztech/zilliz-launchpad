"""Payload-shape tests for the new Phase 5/6 error codes."""

from __future__ import annotations

import json

from lib.errors import (
    BulkImportFailedError,
    CliErrorEnvelope,
    ClusterCreateFailedError,
    DestructiveWithoutConfirmError,
    JudgeUnavailableError,
    QrelsMissingError,
)


def test_qrels_missing_has_remediation():
    err = QrelsMissingError()
    payload = err.to_dict()
    assert payload["code"] == "qrels_missing"
    assert "remediation" in payload
    assert "--qrels" in payload["remediation"]


def test_judge_unavailable_surfaces_env_var():
    err = JudgeUnavailableError(provider="openai", env_var="OPENAI_API_KEY")
    payload = err.to_dict()
    assert payload["code"] == "judge_unavailable"
    assert payload["env_var"] == "OPENAI_API_KEY"
    assert payload["export_hint"].startswith("export OPENAI_API_KEY=")


def test_cluster_create_failed_carries_exit_code_and_stderr():
    err = ClusterCreateFailedError(stderr="quota exceeded", exit_code=2)
    payload = err.to_dict()
    assert payload["code"] == "cluster_create_failed"
    assert payload["stderr"] == "quota exceeded"
    assert payload["exit_code"] == 2


def test_bulk_import_failed_preserves_job_id():
    err = BulkImportFailedError(job_id="job-123", reason="row 42 missing id")
    payload = err.to_dict()
    assert payload["code"] == "bulk_import_failed"
    assert payload["job_id"] == "job-123"
    assert payload["reason"] == "row 42 missing id"


def test_bulk_import_failed_handles_missing_job_id():
    err = BulkImportFailedError(job_id=None, reason="submit rejected")
    payload = err.to_dict()
    assert payload["job_id"] is None
    # The formatted message should degrade gracefully, not crash, when no id
    assert "(no-id)" in payload["message"]


def test_destructive_without_confirm_lists_resources():
    err = DestructiveWithoutConfirmError(
        action="zilliz cluster create", resources=["cluster:my-new-cluster"]
    )
    payload = err.to_dict()
    assert payload["code"] == "destructive_without_confirm"
    assert payload["action"] == "zilliz cluster create"
    assert payload["resources"] == ["cluster:my-new-cluster"]
    assert "remediation" in payload


def test_envelope_roundtrip_keeps_extras():
    """CliErrorEnvelope should preserve code/message and flatten payload extras."""
    err = JudgeUnavailableError(provider="openai", env_var="OPENAI_API_KEY")
    env = CliErrorEnvelope.from_error(err)
    parsed = json.loads(env.to_json())
    assert parsed["code"] == "judge_unavailable"
    assert parsed["env_var"] == "OPENAI_API_KEY"
    # Keys are sorted so the CLI's stderr output is deterministic
    assert list(parsed.keys()) == sorted(parsed.keys())
