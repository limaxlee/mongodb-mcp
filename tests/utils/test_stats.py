import math
import pytest

from mongodb_mcp.utils import stats


class TestDescriptive:
    def test_quantile_interpolates(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert stats.quantile(values, 0.5) == 2.5
        assert stats.quantile(values, 0.0) == 1.0
        assert stats.quantile(values, 1.0) == 4.0
        assert stats.quantile([], 0.5) is None
        assert stats.quantile([7.0], 0.9) == 7.0

    def test_quantiles_keys(self):
        result = stats.quantiles([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
        assert set(result) == {"p10", "p50", "p90"}
        assert result["p50"] == 0.5

        result = stats.quantiles(list(range(1, 11)), [0.25, 0.5, 0.75])
        assert (result["p25"], result["p50"], result["p75"]) == (3.25, 5.5, 7.75)

    def test_histogram_fixed_bins(self):
        edges = [0.0, 0.5, 1.0]
        assert stats.histogram([0.0, 0.25, 0.5, 0.75, 1.0], edges) == [2, 3]
        assert stats.histogram([-1.0, 2.0], edges) == [1, 1]
        assert stats.histogram([], edges) == [0, 0]
        assert stats.proportions([2, 2]) == [0.5, 0.5]
        assert stats.proportions([0, 0]) == [0.0, 0.0]

    def test_mean_std_median(self):
        assert stats.mean([1.0, 3.0]) == 2.0
        assert stats.std([1.0, 3.0]) == 1.0
        assert stats.median([3.0, 1.0, 2.0]) == 2.0
        assert stats.mean([]) is None
        assert stats.std([]) is None
        assert stats.median([]) is None


class TestDivergence:
    def test_psi_identical_is_zero(self):
        assert stats.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)
        assert stats.psi([], []) == 0.0
        assert stats.psi([0.5, 0.5], [1.0]) == 0.0

    def test_psi_known_value_and_symmetry(self):
        assert stats.psi([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.0405, abs=1e-4)
        assert stats.psi([0.9, 0.1], [0.4, 0.6]) > 0.25
        assert stats.psi([0.9, 0.1], [0.4, 0.6]) == pytest.approx(stats.psi([0.4, 0.6], [0.9, 0.1]))

    def test_psi_handles_zero_bins(self):
        value = stats.psi([1.0, 0.0], [0.0, 1.0])
        assert math.isfinite(value) and value > 1.0

    def test_js_bounded(self):
        assert stats.js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
        assert 0.99 < stats.js_divergence([1.0, 0.0], [0.0, 1.0]) <= 1.0

    def test_chi2_known_table(self):
        stat, p, dof = stats.chi2_contingency([[10, 20], [20, 10]])
        assert stat == pytest.approx(6.6667, abs=1e-3)
        assert p == pytest.approx(0.00982, abs=1e-4)
        assert dof == 1

        stat, p, dof = stats.chi2_contingency([[30, 70], [50, 50]])
        assert stat == pytest.approx(8.3333, abs=1e-3)
        assert p == pytest.approx(0.00389, abs=1e-4)

    def test_chi2_degenerate_tables(self):
        assert stats.chi2_contingency([[10, 0], [20, 0]]) == (0.0, 1.0, 0)
        assert stats.chi2_contingency([]) == (0.0, 1.0, 0)

    def test_chi2_survival_matches_reference(self):
        assert stats.chi2_survival(3.841, 1) == pytest.approx(0.05, abs=1e-3)
        assert stats.chi2_survival(5.991, 2) == pytest.approx(0.05, abs=1e-3)
        assert stats.chi2_survival(9.488, 4) == pytest.approx(0.05, abs=1e-3)
        assert stats.chi2_survival(10.828, 1) == pytest.approx(0.001, abs=1e-4)
        assert stats.chi2_survival(0.0, 3) == 1.0


class TestHistogramQuantile:
    def test_median_inside_a_bin(self):
        # 10 values uniform in [0, 1): the median is halfway through the histogram
        edges = [0.0, 0.5, 1.0]
        assert stats.histogram_quantile([5, 5], edges, 0.5) == 0.5
        assert stats.histogram_quantile([10, 0], edges, 0.5) == 0.25
        assert stats.histogram_quantile([0, 10], edges, 0.5) == 0.75

    def test_quantile_positions(self):
        edges = [0.0, 0.1, 0.2, 0.3]
        counts = [1, 1, 2]
        assert math.isclose(stats.histogram_quantile(counts, edges, 0.25), 0.1)
        assert math.isclose(stats.histogram_quantile(counts, edges, 0.5), 0.2)
        assert math.isclose(stats.histogram_quantile(counts, edges, 1.0), 0.3)

    def test_degenerate(self):
        assert stats.histogram_quantile([], [0.0, 1.0], 0.5) is None
        assert stats.histogram_quantile([0, 0], [0.0, 0.5, 1.0], 0.5) is None
        assert stats.histogram_quantile([1, 2, 3], [0.0, 1.0], 0.5) is None
