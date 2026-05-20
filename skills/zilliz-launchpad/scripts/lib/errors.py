"""Structured error classes with a common CLI-serializable form.

All launchpad errors inherit `LaunchpadError` so the CLI can catch a single
type and render a consistent JSON payload — agents parse this payload to
decide whether to prompt the user, retry, or abort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class LaunchpadError(Exception):
    """Base class. Subclasses fill `code` and `payload`."""

    code: str = "launchpad_error"

    def __init__(self, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.message = message
        self.payload: dict[str, Any] = payload

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class MissingCredentialError(LaunchpadError):
    code = "missing_credential"

    def __init__(self, env_var: str, *, hint: str | None = None) -> None:
        super().__init__(
            f"Required credential not set: {env_var}",
            env_var=env_var,
            export_hint=hint or f"export {env_var}=<value>",
        )


class SchemaConflictError(LaunchpadError):
    code = "schema_conflict"

    def __init__(self, collection: str, mismatches: list[str]) -> None:
        super().__init__(
            f"Existing collection '{collection}' has incompatible schema",
            collection=collection,
            mismatches=mismatches,
        )


class InputSchemaConflictError(LaunchpadError):
    """Two or more input files declare the same field name with different JSON types.

    Distinct from `SchemaConflictError`, which covers live-collection vs `plan.json`
    mismatches at execute time.
    """

    code = "input_schema_conflict"

    def __init__(self, *, field_name: str, files_and_types: list[dict[str, str]]) -> None:
        files_desc = ", ".join(f"{ft['path']}({ft['type']})" for ft in files_and_types)
        super().__init__(
            f"Field '{field_name}' has conflicting types across inputs: {files_desc}",
            field=field_name,
            files=files_and_types,
        )


class EmptyInputSetError(LaunchpadError):
    """`--input` resolved to zero files."""

    code = "empty_input"

    def __init__(self, *, raw: str, reason: str) -> None:
        super().__init__(
            f"--input '{raw}' resolved to no files: {reason}",
            raw=raw,
            reason=reason,
        )


class SparseUnavailable(LaunchpadError):
    code = "sparse_unavailable"

    def __init__(self, collection: str) -> None:
        super().__init__(
            f"Collection '{collection}' has no sparse vector field",
            collection=collection,
            remediation="Re-ingest with `sparse.enabled=true` in the plan.",
        )


class InvalidProfileError(LaunchpadError):
    code = "invalid_profile"

    def __init__(self, pointer: str, reason: str) -> None:
        super().__init__(
            f"Requirement profile invalid at {pointer}: {reason}",
            pointer=pointer,
            reason=reason,
        )


class BackendUnsupportedError(LaunchpadError):
    code = "backend_unsupported"

    def __init__(self, uri: str, feature: str) -> None:
        super().__init__(
            f"Backend at {uri} does not support {feature}",
            uri=uri,
            feature=feature,
        )


ZILLIZ_CLI_INSTALL_URL = "https://github.com/zilliztech/zilliz-cli#installation"


class ZillizCliMissingError(LaunchpadError):
    code = "zilliz_cli_missing"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "zilliz CLI is required for this operation but was not found on PATH",
            install_url=ZILLIZ_CLI_INSTALL_URL,
        )


class ZillizCliAuthError(LaunchpadError):
    code = "zilliz_cli_auth"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "zilliz CLI is installed but not authenticated",
            remediation="zilliz auth login",
        )


class ClusterNotReadyError(LaunchpadError):
    code = "cluster_not_ready"

    def __init__(self, *, cluster_id: str, state: str, remediation: str) -> None:
        super().__init__(
            f"Cluster {cluster_id} is not ready (state={state})",
            cluster_id=cluster_id,
            state=state,
            remediation=remediation,
        )


class QrelsMissingError(LaunchpadError):
    code = "qrels_missing"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "Comparison mode requires labelled qrels to rank variants",
            remediation="Pass --qrels <path> with {query, relevant_ids[]} per line",
        )


class JudgeUnavailableError(LaunchpadError):
    code = "judge_unavailable"

    def __init__(self, *, provider: str, env_var: str) -> None:
        super().__init__(
            f"Judge LLM '{provider}' requested but no credential is configured",
            provider=provider,
            env_var=env_var,
            export_hint=f"export {env_var}=<value>",
        )


class ClusterCreateFailedError(LaunchpadError):
    code = "cluster_create_failed"

    def __init__(self, *, stderr: str, exit_code: int) -> None:
        super().__init__(
            "zilliz cluster create failed",
            stderr=stderr,
            exit_code=exit_code,
        )


class BulkImportFailedError(LaunchpadError):
    code = "bulk_import_failed"

    def __init__(self, *, job_id: str | None, reason: str) -> None:
        super().__init__(
            f"zilliz import job {job_id or '(no-id)'} failed: {reason}",
            job_id=job_id,
            reason=reason,
        )


class MissingDependencyError(LaunchpadError):
    code = "missing_dependency"

    def __init__(self, *, feature: str, install_hint: str) -> None:
        super().__init__(
            f"Optional dependency required for '{feature}' is not installed",
            feature=feature,
            install_hint=install_hint,
        )


class DestructiveWithoutConfirmError(LaunchpadError):
    code = "destructive_without_confirm"

    def __init__(self, *, action: str, resources: list[str]) -> None:
        super().__init__(
            f"Refusing to run destructive action '{action}' without --confirm",
            action=action,
            resources=resources,
            remediation="Re-run with --confirm after reviewing projected impact",
        )


class UnsupportedImageProviderError(LaunchpadError):
    code = "unsupported_image_provider"

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Embedding provider '{provider}' has no image modality; "
            "re-run plan with an image-capable provider (clip-local or voyage-multimodal-3)",
            provider=provider,
        )


class ImageDecodeError(LaunchpadError):
    code = "image_decode_failed"

    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "gif"]

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Could not decode upload as a supported image: {reason}",
            reason=reason,
            supported_formats=list(self.SUPPORTED_FORMATS),
        )


@dataclass
class CliErrorEnvelope:
    """Uniform envelope emitted to stderr on failure."""

    code: str
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, err: LaunchpadError) -> CliErrorEnvelope:
        d = err.to_dict()
        code = d.pop("code")
        message = d.pop("message")
        return cls(code=code, message=message, extra=d)

    def to_json(self) -> str:
        return json.dumps(
            {"code": self.code, "message": self.message, **self.extra},
            ensure_ascii=False,
            sort_keys=True,
        )
