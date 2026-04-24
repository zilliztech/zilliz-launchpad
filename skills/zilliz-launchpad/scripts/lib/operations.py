"""Collection and index lifecycle operations.

All operations are idempotent: same input → same end state.
Schema or index-param mismatches raise structured errors rather than
silently overwriting existing work.
"""

from __future__ import annotations

from typing import Any, Literal

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient

from .errors import BackendUnsupportedError, SchemaConflictError

IndexType = Literal["FLAT", "IVF_FLAT", "HNSW", "DISKANN", "SPARSE_INVERTED_INDEX"]
MetricType = Literal["L2", "IP", "COSINE", "BM25"]


def collection_exists(client: MilvusClient, name: str) -> bool:
    return name in client.list_collections()


def _dtype_name(v: Any) -> str:
    """Normalize a pymilvus dtype (enum / int / 'DataType.X' / 'X') to its enum name."""
    if isinstance(v, DataType):
        return str(v.name)
    if isinstance(v, int):
        try:
            return str(DataType(v).name)
        except ValueError:
            return str(v)
    s = str(v)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    if s.isdigit():
        try:
            return str(DataType(int(s)).name)
        except ValueError:
            return s
    return s


def _schema_fingerprint(schema: CollectionSchema) -> list[tuple[str, str, dict[str, Any]]]:
    """Stable representation of a schema for comparison."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for f in schema.fields:
        params = dict(getattr(f, "params", {}) or {})
        out.append((f.name, _dtype_name(f.dtype), params))
    return sorted(out, key=lambda r: r[0])


def _classify_diff(
    existing_fields: list[dict[str, Any]],
    requested: CollectionSchema,
) -> tuple[list[FieldSchema], list[str]]:
    """Split a schema diff into (additive, fatal).

    `additive` is the list of FieldSchema entries present in the plan but not
    in the existing collection — candidates for `add_collection_field`.
    `fatal` is a list of human-readable strings describing incompatible
    differences (type/param mismatch, or fields present in the existing
    collection but dropped from the plan).
    """
    have = {
        f["name"]: (_dtype_name(f.get("type")), dict(f.get("params") or {}))
        for f in existing_fields
    }
    wanted_names = {f.name for f in requested.fields}

    additive: list[FieldSchema] = []
    fatal: list[str] = []

    for f in requested.fields:
        wtype = _dtype_name(f.dtype)
        wparams = dict(getattr(f, "params", {}) or {})
        if f.name not in have:
            additive.append(f)
            continue
        htype, hparams = have[f.name]
        if wtype != htype:
            fatal.append(f"field '{f.name}' type differs: have {htype}, want {wtype}")
        for key in ("dim", "max_length"):
            if key in wparams and wparams[key] != hparams.get(key):
                fatal.append(
                    f"field '{f.name}' param '{key}' differs: "
                    f"have {hparams.get(key)}, want {wparams[key]}"
                )

    for name in have:
        if name not in wanted_names:
            fatal.append(f"extra field (not in plan): {name}")

    return additive, fatal


def _diff_schemas(
    existing_fields: list[dict[str, Any]],
    requested: CollectionSchema,
) -> list[str]:
    """Return human-readable mismatch strings; empty list if compatible.

    Preserved as a thin wrapper over `_classify_diff` so callers that just
    want a flat list (tests, error payloads) keep working.
    """
    additive, fatal = _classify_diff(existing_fields, requested)
    return [f"missing field: {f.name}" for f in additive] + fatal


def _add_field_kwargs(field: FieldSchema) -> dict[str, Any]:
    """Build kwargs for `MilvusClient.add_collection_field` from a FieldSchema.

    Fields added to a live collection must be nullable — Milvus fills existing
    rows with null / default so the operation stays online.
    """
    params = dict(getattr(field, "params", {}) or {})
    kwargs: dict[str, Any] = {"nullable": True}
    # Forward only the field-schema kwargs that `create_field_schema` accepts.
    for key in ("max_length", "dim", "element_type", "max_capacity"):
        if key in params:
            kwargs[key] = params[key]
    return kwargs


def create_collection(
    client: MilvusClient,
    name: str,
    schema: CollectionSchema,
) -> Literal["created", "reused", "extended"]:
    """Idempotent create with additive schema evolution.

    - If the collection is absent: create it.
    - If present and schema matches: reuse.
    - If present and the only differences are fields the plan added that the
      collection lacks (type/param of shared fields still match): call
      `add_collection_field` for each and return `"extended"`. The new fields
      are added as nullable so existing rows remain valid.
    - Otherwise: raise `SchemaConflictError`.
    """
    if not collection_exists(client, name):
        client.create_collection(collection_name=name, schema=schema)
        return "created"

    info = client.describe_collection(name)
    existing_fields = info.get("fields", [])
    additive, fatal = _classify_diff(existing_fields, schema)
    if fatal:
        mismatches = [f"missing field: {f.name}" for f in additive] + fatal
        raise SchemaConflictError(collection=name, mismatches=mismatches)
    if not additive:
        return "reused"
    for field in additive:
        client.add_collection_field(
            collection_name=name,
            field_name=field.name,
            data_type=field.dtype,
            **_add_field_kwargs(field),
        )
    return "extended"


def load_collection(client: MilvusClient, name: str) -> None:
    client.load_collection(collection_name=name)


def drop_collection(client: MilvusClient, name: str) -> None:
    if collection_exists(client, name):
        client.drop_collection(collection_name=name)


def describe_index(client: MilvusClient, collection: str, field_name: str) -> dict[str, Any] | None:
    for idx in client.list_indexes(collection_name=collection) or []:
        info: dict[str, Any] = client.describe_index(collection_name=collection, index_name=idx)
        if info.get("field_name") == field_name:
            return info
    return None


def _index_params_equal(have: dict[str, Any], want: dict[str, Any]) -> bool:
    def norm(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v not in (None, "", {})}

    return norm(have) == norm(want)


def create_index(
    client: MilvusClient,
    collection: str,
    field_name: str,
    index_type: IndexType,
    metric_type: MetricType,
    params: dict[str, Any] | None = None,
) -> Literal["created", "reused", "rebuilt"]:
    """Idempotent create / drop-and-rebuild on param mismatch."""
    params = params or {}
    wanted = {"index_type": index_type, "metric_type": metric_type, "params": params}

    existing = describe_index(client, collection, field_name)
    if existing is not None:
        have = {
            "index_type": existing.get("index_type"),
            "metric_type": existing.get("metric_type"),
            "params": existing.get("params") or {},
        }
        if _index_params_equal(have, wanted):
            return "reused"
        # drop and rebuild
        client.drop_index(collection_name=collection, index_name=existing.get("index_name", ""))

    try:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_type,
            metric_type=metric_type,
            params=params,
        )
        client.create_index(collection_name=collection, index_params=index_params)
    except Exception as e:  # pymilvus raises a raw Exception for unsupported indexes
        if "not support" in str(e).lower() or "unsupported" in str(e).lower():
            raise BackendUnsupportedError(uri=str(client._using), feature=index_type) from e
        raise

    return "rebuilt" if existing is not None else "created"


def drop_index(client: MilvusClient, collection: str, field_name: str) -> None:
    existing = describe_index(client, collection, field_name)
    if existing is not None:
        client.drop_index(collection_name=collection, index_name=existing.get("index_name", ""))


def build_basic_schema(
    *,
    primary_field: str = "id",
    text_field: str | None = "text",
    vector_field: str = "embedding",
    dim: int,
    enable_sparse: bool = False,
    sparse_field: str = "sparse",
    primary_max_length: int = 128,
    extra_fields: list[tuple[str, DataType, int | None]] | None = None,
) -> CollectionSchema:
    """Build a basic schema shared by the plan-generated collection shape.

    `text_field=None` builds a schema without a text column — used by image
    collections where the primary key carries the file path and there's no
    indexed natural-language column. `enable_sparse=True` requires text_field.

    `extra_fields` is a list of `(name, dtype, max_length_or_none)`.
    """
    if enable_sparse and text_field is None:
        raise ValueError("enable_sparse=True requires a text_field for the BM25 function input")

    fields: list[FieldSchema] = [
        FieldSchema(
            name=primary_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=primary_max_length,
        ),
    ]
    if text_field is not None:
        text_field_kwargs: dict[str, Any] = {"max_length": 65535}
        if enable_sparse:
            text_field_kwargs["enable_analyzer"] = True
        fields.append(FieldSchema(name=text_field, dtype=DataType.VARCHAR, **text_field_kwargs))
    fields.append(FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim))
    if enable_sparse:
        fields.append(FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR))
    for name, dtype, max_length in extra_fields or []:
        if dtype == DataType.VARCHAR and max_length is not None:
            fields.append(FieldSchema(name=name, dtype=dtype, max_length=max_length))
        else:
            fields.append(FieldSchema(name=name, dtype=dtype))
    schema = CollectionSchema(fields=fields, description="zilliz-launchpad collection")
    if enable_sparse and text_field is not None:
        schema.add_function(
            Function(
                name=f"{text_field}_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[text_field],
                output_field_names=[sparse_field],
            )
        )
    return schema
