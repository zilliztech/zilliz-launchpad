"""Collection and index lifecycle operations.

All operations are idempotent: same input → same end state.
Schema or index-param mismatches raise structured errors rather than
silently overwriting existing work.
"""

from __future__ import annotations

from typing import Any, Literal

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient

from .errors import BackendUnsupportedError, SchemaConflictError

IndexType = Literal["FLAT", "IVF_FLAT", "HNSW", "DISKANN"]


def collection_exists(client: MilvusClient, name: str) -> bool:
    return name in client.list_collections()


def _dtype_name(v: Any) -> str:
    """Normalize a pymilvus dtype (enum / int / 'DataType.X' / 'X') to its enum name."""
    if isinstance(v, DataType):
        return v.name
    if isinstance(v, int):
        try:
            return DataType(v).name
        except ValueError:
            return str(v)
    s = str(v)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    if s.isdigit():
        try:
            return DataType(int(s)).name
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


def _diff_schemas(
    existing_fields: list[dict[str, Any]],
    requested: CollectionSchema,
) -> list[str]:
    """Return human-readable mismatch strings; empty list if compatible."""
    wanted = {
        f.name: (_dtype_name(f.dtype), dict(getattr(f, "params", {}) or {}))
        for f in requested.fields
    }
    have = {
        f["name"]: (_dtype_name(f.get("type")), dict(f.get("params") or {}))
        for f in existing_fields
    }

    mismatches: list[str] = []
    for name, (wtype, wparams) in wanted.items():
        if name not in have:
            mismatches.append(f"missing field: {name}")
            continue
        htype, hparams = have[name]
        if wtype != htype:
            mismatches.append(f"field '{name}' type differs: have {htype}, want {wtype}")
        for key in ("dim", "max_length"):
            if key in wparams and wparams[key] != hparams.get(key):
                mismatches.append(
                    f"field '{name}' param '{key}' differs: "
                    f"have {hparams.get(key)}, want {wparams[key]}"
                )
    for name in have:
        if name not in wanted:
            mismatches.append(f"extra field (not in plan): {name}")
    return mismatches


def create_collection(
    client: MilvusClient,
    name: str,
    schema: CollectionSchema,
) -> Literal["created", "reused"]:
    """Idempotent create. Raises `SchemaConflictError` on mismatch."""
    if collection_exists(client, name):
        info = client.describe_collection(name)
        existing_fields = info.get("fields", [])
        mismatches = _diff_schemas(existing_fields, schema)
        if mismatches:
            raise SchemaConflictError(collection=name, mismatches=mismatches)
        return "reused"
    client.create_collection(collection_name=name, schema=schema)
    return "created"


def load_collection(client: MilvusClient, name: str) -> None:
    client.load_collection(collection_name=name)


def drop_collection(client: MilvusClient, name: str) -> None:
    if collection_exists(client, name):
        client.drop_collection(collection_name=name)


def describe_index(client: MilvusClient, collection: str, field_name: str) -> dict[str, Any] | None:
    for idx in client.list_indexes(collection_name=collection) or []:
        info = client.describe_index(collection_name=collection, index_name=idx)
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
    metric_type: Literal["L2", "IP", "COSINE"],
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
    text_field: str = "text",
    vector_field: str = "embedding",
    dim: int,
    enable_sparse: bool = False,
    sparse_field: str = "sparse",
    extra_fields: list[tuple[str, DataType, int | None]] | None = None,
) -> CollectionSchema:
    """Build a basic schema shared by the plan-generated collection shape.

    `extra_fields` is a list of `(name, dtype, max_length_or_none)`.
    """
    text_field_kwargs: dict[str, Any] = {"max_length": 65535}
    if enable_sparse:
        text_field_kwargs["enable_analyzer"] = True
    fields: list[FieldSchema] = [
        FieldSchema(name=primary_field, dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name=text_field, dtype=DataType.VARCHAR, **text_field_kwargs),
        FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    if enable_sparse:
        fields.append(FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR))
    for name, dtype, max_length in extra_fields or []:
        if dtype == DataType.VARCHAR and max_length is not None:
            fields.append(FieldSchema(name=name, dtype=dtype, max_length=max_length))
        else:
            fields.append(FieldSchema(name=name, dtype=dtype))
    schema = CollectionSchema(fields=fields, description="zilliz-launchpad collection")
    if enable_sparse:
        schema.add_function(
            Function(
                name=f"{text_field}_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[text_field],
                output_field_names=[sparse_field],
            )
        )
    return schema
