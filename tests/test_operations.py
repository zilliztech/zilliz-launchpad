"""Schema-diff behavior for `lib.operations`."""

from __future__ import annotations

import pytest
from lib.errors import SchemaConflictError
from lib.operations import _diff_schemas, create_collection
from pymilvus import CollectionSchema, DataType, FieldSchema


def _schema(dim: int = 4) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
    )


def test_diff_identical_schemas_empty():
    s = _schema(dim=4)
    existing = [
        {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
        {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
        {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
    ]
    assert _diff_schemas(existing, s) == []


def test_diff_dim_mismatch_surfaces():
    s = _schema(dim=8)
    existing = [
        {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
        {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
        {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
    ]
    diffs = _diff_schemas(existing, s)
    assert any("dim" in d for d in diffs)


def test_diff_missing_field():
    s = _schema()
    existing = [
        {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
        {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
    ]
    diffs = _diff_schemas(existing, s)
    assert any("missing field" in d for d in diffs)


def test_create_collection_raises_on_conflict(mocker):
    client = mocker.Mock()
    client.list_collections.return_value = ["docs"]
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
            {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 999}},
        ]
    }
    with pytest.raises(SchemaConflictError) as exc:
        create_collection(client, "docs", _schema(dim=4))
    assert exc.value.payload["collection"] == "docs"


def test_create_collection_reuses_on_match(mocker):
    client = mocker.Mock()
    client.list_collections.return_value = ["docs"]
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
            {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
        ]
    }
    status = create_collection(client, "docs", _schema(dim=4))
    assert status == "reused"
    client.create_collection.assert_not_called()
