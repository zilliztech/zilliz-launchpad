"""Phase 3 Plan — video-search plan."""

from __future__ import annotations

from lib.phases.plan import plan_from_profile


def _profile(**configure_overrides: object) -> dict[str, object]:
    configure: dict[str, object] = {
        "use_case": "video-search",
        "dataset_size": 25,
        "deployment_target": "local-standalone",
        "frame_interval_seconds": 2.0,
        "max_frames_per_video": 600,
        "sampling_strategy": "every_n_seconds",
        "scene_threshold": 0.3,
    }
    configure.update(configure_overrides)
    collect: dict[str, object] = {
        "data_shape": "video_dir",
        "suggested_primary_key": "frame_path",
        "suggested_text_field": None,
        "record_count_estimate": 25,
        "video_count": 5,
        "fields": [
            {"name": "video_path", "type": "string"},
            {"name": "t_seconds", "type": "float"},
            {"name": "frame_path", "type": "string"},
            {"name": "source_index", "type": "int"},
        ],
    }
    return {"collect": collect, "configure": configure}


def test_default_clip_local_video_plan():
    plan = plan_from_profile(_profile())
    assert plan.embedding["provider"] == "clip-local"
    assert plan.embedding["model"] == "ViT-B-32"
    assert plan.embedding["dim"] == 512
    assert plan.embedding.get("modality") == "image"
    assert plan.sparse_enabled is False
    assert plan.reranker is None
    # Schema carries the deep-link scalars
    names = {f["name"] for f in plan.schema["extra_fields"]}
    assert {"video_path", "t_seconds"} <= names
    video_path_field = next(f for f in plan.schema["extra_fields"] if f["name"] == "video_path")
    assert video_path_field["max_length"] == 1024
    # Chunking includes the video section
    assert plan.chunking["video"]["frame_interval_seconds"] == 2.0
    assert plan.chunking["video"]["max_frames_per_video"] == 600
    assert plan.chunking["video"]["sampling_strategy"] == "every_n_seconds"
    assert plan.schema["is_video"] is True


def test_voyage_override_for_video():
    plan = plan_from_profile(_profile(embedding_preference="voyage-multimodal-3"))
    assert plan.embedding["provider"] == "voyage"
    assert plan.embedding["model"] == "voyage-multimodal-3"
    assert plan.embedding["dim"] == 1024


def test_plan_md_contains_video_section():
    from lib.phases.plan import _plan_to_markdown

    plan = plan_from_profile(_profile())
    md = _plan_to_markdown(plan)
    assert "## Video" in md
    assert "Frame interval: 2.0 s" in md
    assert "Max frames per video: 600" in md
    assert "Scalar fields" in md
    assert "video_path" in md
    assert "Sparse field: disabled (video collection)" in md
    assert "Text field: (none — video collection)" in md


def test_plan_md_scene_change_section():
    from lib.phases.plan import _plan_to_markdown

    plan = plan_from_profile(_profile(sampling_strategy="scene_change", scene_threshold=0.5))
    md = _plan_to_markdown(plan)
    assert "scene_change" in md
    assert "Scene threshold: 0.5" in md


def test_sparse_disabled_in_schema_for_video():
    plan = plan_from_profile(_profile())
    assert plan.schema["sparse_field"] is None
    assert plan.sparse_enabled is False


def test_voyage_cost_line_in_rationale():
    plan = plan_from_profile(_profile(embedding_preference="voyage-multimodal-3"))
    cost_lines = [r for r in plan.rationale if "cost" in r.lower()]
    assert cost_lines, plan.rationale
