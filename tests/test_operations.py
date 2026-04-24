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


def _schema_with_tags(dim: int = 4) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512),
        ]
    )


def test_create_collection_extends_on_missing_field(mocker):
    client = mocker.Mock()
    client.list_collections.return_value = ["docs"]
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
            {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
        ]
    }
    status = create_collection(client, "docs", _schema_with_tags(dim=4))
    assert status == "extended"
    client.create_collection.assert_not_called()
    client.drop_collection.assert_not_called()
    client.add_collection_field.assert_called_once()
    _, kwargs = client.add_collection_field.call_args
    assert kwargs["collection_name"] == "docs"
    assert kwargs["field_name"] == "tags"
    assert kwargs["data_type"] == DataType.VARCHAR
    assert kwargs["nullable"] is True
    assert kwargs["max_length"] == 512


def test_create_collection_still_raises_on_type_mismatch(mocker):
    """Additive path must not mask genuine incompatibilities."""
    client = mocker.Mock()
    client.list_collections.return_value = ["docs"]
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
            {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 999}},
        ]
    }
    with pytest.raises(SchemaConflictError):
        create_collection(client, "docs", _schema_with_tags(dim=4))
    client.add_collection_field.assert_not_called()


def test_create_collection_raises_on_field_dropped_from_plan(mocker):
    """A field present in the live collection but missing from the new plan
    is destructive — keep it fatal."""
    client = mocker.Mock()
    client.list_collections.return_value = ["docs"]
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
            {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
            {"name": "legacy", "type": "DataType.VARCHAR", "params": {"max_length": 32}},
        ]
    }
    with pytest.raises(SchemaConflictError) as exc:
        create_collection(client, "docs", _schema(dim=4))
    assert any("legacy" in m for m in exc.value.payload["mismatches"])
    client.add_collection_field.assert_not_called()
