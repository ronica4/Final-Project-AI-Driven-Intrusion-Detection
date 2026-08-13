"""
Step 2B verification. Synthetic data only, hand-computed where possible --
same discipline as tests/test_metrics.py (real math checked against math
done independently of the library under test, not sklearn/scipy calling
itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.selection import (
    cliffs_delta_from_u,
    class_distribution_counts,
    correlation_heatmap,
    effect_size_verdict,
    feature_significance_table,
    near_constant_report,
    plot_class_distribution,
    plot_feature_distributions,
    three_way_breakdown,
)


def test_cliffs_delta_complete_separation_is_hand_computed_one():
    # attack strictly greater than benign in every one of the 5*5=25 pairs --
    # U is exactly 25 by hand (no need to trust scipy's internals for this
    # degenerate case), so delta = 2*25/25 - 1 = 1.0 exactly.
    benign = [1, 2, 3, 4, 5]
    attack = [6, 7, 8, 9, 10]
    from scipy import stats

    u_stat, p_value = stats.mannwhitneyu(attack, benign, alternative="two-sided")
    assert u_stat == pytest.approx(25.0)
    delta = cliffs_delta_from_u(u_stat, len(attack), len(benign))
    assert delta == pytest.approx(1.0)
    assert effect_size_verdict(abs(delta)) == "large"
    assert p_value < 0.01  # n=5,5 exact test; complete separation is highly significant


def test_cliffs_delta_identical_distributions_is_hand_computed_zero():
    # Identical groups -> by symmetry every tie contributes 0.5 both ways,
    # so U = n1*n2/2 = 4.5 exactly, delta = 2*4.5/9 - 1 = 0.
    from scipy import stats

    x = [1, 2, 3]
    y = [1, 2, 3]
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    assert u_stat == pytest.approx(4.5)
    delta = cliffs_delta_from_u(u_stat, len(x), len(y))
    assert delta == pytest.approx(0.0)
    assert effect_size_verdict(abs(delta)) == "negligible"


@pytest.mark.parametrize(
    "delta,expected",
    [(0.05, "negligible"), (0.2, "small"), (0.4, "medium"), (0.6, "large")],
)
def test_effect_size_verdict_bands(delta, expected):
    assert effect_size_verdict(delta) == expected


def test_class_distribution_counts_matches_hand_count():
    y = [0, 0, 0, 1, 1]
    counts = class_distribution_counts(y)
    assert counts == {"benign": 3, "attack": 2}


def _synthetic_two_class_frame(n=200, random_state=42):
    rng = np.random.default_rng(random_state)
    benign_n = n // 2
    attack_n = n - benign_n
    X = pd.DataFrame(
        {
            "separated": np.concatenate(
                [rng.normal(0, 1, benign_n), rng.normal(5, 1, attack_n)]
            ),
            "all_nan": np.full(n, np.nan),
            "constant": np.full(n, 7.0),
        }
    )
    y = pd.Series([0] * benign_n + [1] * attack_n)
    return X, y


def test_feature_significance_table_flags_untestable_and_real_separation():
    X, y = _synthetic_two_class_frame()
    rows = feature_significance_table(X, y)
    by_name = {r["feature"]: r for r in rows}

    assert by_name["all_nan"]["verdict"].startswith("untestable")
    assert by_name["all_nan"]["cliffs_delta"] is None

    # "separated" was constructed with benign~N(0,1), attack~N(5,1) -- a
    # huge, unambiguous effect; verdict must be "large" and p tiny.
    assert by_name["separated"]["verdict"] == "large"
    assert by_name["separated"]["p_value"] < 1e-10
    assert by_name["separated"]["median_attack"] > by_name["separated"]["median_benign"]


def test_correlation_heatmap_flags_perfectly_correlated_pair(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": a,
            "b": a * 2.0 + 1.0,  # perfectly correlated with a (r=1.0)
            "c": rng.normal(size=n),  # independent
        }
    )
    corr, flagged = correlation_heatmap(X, "synthetic", tmp_path / "corr.png", threshold=0.9)

    assert corr.loc["a", "b"] == pytest.approx(1.0)
    flagged_pairs = {frozenset([f, s]) for f, s, _ in flagged}
    assert frozenset(["a", "b"]) in flagged_pairs
    assert frozenset(["a", "c"]) not in flagged_pairs
    assert (tmp_path / "corr.png").exists()


def test_three_way_breakdown_light_closer_to_benign_than_heavy():
    """Construct light to sit numerically closer to benign than heavy does,
    and confirm the two Cliff's deltas reflect that ordering -- this is the
    exact claim Step 3A's error analysis depends on."""
    rng = np.random.default_rng(1)
    n = 100
    X = pd.DataFrame(
        {
            "vol": np.concatenate(
                [
                    rng.normal(0, 1, n),  # benign
                    rng.normal(1, 1, n),  # light -- close to benign
                    rng.normal(10, 1, n),  # heavy -- far from benign
                ]
            )
        }
    )
    attack_subclass = pd.Series(["benign"] * n + ["light_attack"] * n + ["heavy_attack"] * n)

    rows = three_way_breakdown(X, attack_subclass)
    row = rows[0]
    assert row["kruskal_p"] < 1e-10
    assert abs(row["cliffs_delta_light_vs_benign"]) < abs(row["cliffs_delta_heavy_vs_benign"])


def test_near_constant_report_flags_constant_and_all_nan_columns():
    X, _ = _synthetic_two_class_frame()
    report = {r["feature"]: r for r in near_constant_report(X)}

    assert report["constant"]["near_constant"] is True
    assert report["constant"]["nan_fraction"] == pytest.approx(0.0)
    assert report["all_nan"]["near_constant"] is True
    assert report["all_nan"]["nan_fraction"] == pytest.approx(1.0)
    assert report["separated"]["near_constant"] is False


def test_plot_class_distribution_writes_a_file(tmp_path):
    path = tmp_path / "dist.png"
    plot_class_distribution({"benign": 100, "attack": 20}, "test", path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_feature_distributions_writes_one_file_per_nonempty_column(tmp_path):
    X, y = _synthetic_two_class_frame()
    written = plot_feature_distributions(X, y, tmp_path, "synthetic")
    written_names = {p.name for p in written}
    # "all_nan" has nothing to plot for either class -- must be skipped.
    assert not any("all_nan" in name for name in written_names)
    assert any("separated" in name for name in written_names)
    assert any("constant" in name for name in written_names)
    for p in written:
        assert p.exists()
