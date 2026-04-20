from __future__ import annotations

from lib.search import Hit, _rrf, _weighted


def _hits(ids: list[str], scores: list[float] | None = None) -> list[Hit]:
    if scores is None:
        scores = [1.0] * len(ids)
    return [Hit(id=i, score=s) for i, s in zip(ids, scores, strict=True)]


def test_rrf_boosts_items_present_in_both_lists():
    dense = _hits(["A", "B", "C"])
    sparse = _hits(["B", "D", "E"])
    fused = _rrf([dense, sparse])
    top = [h.id for h in fused]
    assert top.index("B") < top.index("A")
    assert top.index("B") < top.index("D")


def test_weighted_fusion_respects_weights():
    dense = _hits(["A", "B"], [10.0, 1.0])
    sparse = _hits(["C", "A"], [10.0, 1.0])
    # Heavily bias dense → A is boosted; small sparse weight
    fused = _weighted([dense, sparse], weights=(0.9, 0.1))
    assert fused[0].id == "A"


def test_fusion_preserves_hit_metadata():
    dense = [Hit(id="x", score=2.0, fields={"text": "hello"})]
    sparse: list[Hit] = []
    fused = _rrf([dense, sparse])
    assert fused[0].fields["text"] == "hello"
