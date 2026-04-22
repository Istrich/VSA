"""Unit tests for ``HybridSearchEngine`` scoring and normalization (BUG-N17/N18)."""

from __future__ import annotations

import pytest

from core.models import SearchWeights
from core.search import HybridSearchEngine


class TestNormalizePairs:
    def test_empty_input(self) -> None:
        assert HybridSearchEngine._normalize_pairs([]) == []

    def test_single_positive_value_returns_one(self) -> None:
        # A single candidate must not collapse to score=0 (BUG-N17).
        assert HybridSearchEngine._normalize_pairs([0.42]) == [1.0]

    def test_single_zero_value_returns_zero(self) -> None:
        assert HybridSearchEngine._normalize_pairs([0.0]) == [0.0]

    def test_all_equal_positive_values(self) -> None:
        assert HybridSearchEngine._normalize_pairs([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]

    def test_all_equal_zero_values(self) -> None:
        assert HybridSearchEngine._normalize_pairs([0.0, 0.0]) == [0.0, 0.0]

    def test_min_max_linear_interpolation(self) -> None:
        out = HybridSearchEngine._normalize_pairs([1.0, 2.0, 4.0])
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)
        assert out[1] == pytest.approx(1.0 / 3.0)


class TestFinalScoreWeighting:
    def test_weighted_sum_without_renormalization(self) -> None:
        weights = SearchWeights(clip=0.5, face=0.3, fts=0.2)
        candidates = {
            "/a": {"clip_sim": 1.0, "face_sim": 0.5, "fts_score": 0.0, "path": "/a"},
            "/b": {"clip_sim": 0.0, "face_sim": 0.0, "fts_score": 1.0, "path": "/b"},
        }
        HybridSearchEngine._compute_final_scores(candidates, weights)
        assert candidates["/a"]["score"] == pytest.approx(0.5 * 1.0 + 0.3 * 0.5)
        assert candidates["/b"]["score"] == pytest.approx(0.2 * 1.0)

    def test_single_branch_candidate_keeps_weight(self) -> None:
        # Regression for BUG-N18: a candidate that only came through FTS
        # must preserve its FTS contribution; the dormant clip/face slots
        # stay at 0 and therefore contribute 0 to the final score.
        weights = SearchWeights(clip=0.4, face=0.4, fts=0.2)
        candidates = {
            "/only-fts": {
                "clip_sim": 0.0,
                "face_sim": 0.0,
                "fts_score": 1.0,
                "path": "/only-fts",
            }
        }
        HybridSearchEngine._compute_final_scores(candidates, weights)
        assert candidates["/only-fts"]["score"] == pytest.approx(0.2)
