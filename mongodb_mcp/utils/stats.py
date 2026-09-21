"""Pure statistical helpers, no third party dependencies

Every function takes plain Python sequences and returns plain floats so it can be unit tested with synthetic data.
"""
import math
from typing import Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def median(values: Sequence[float]) -> float | None:
    return quantile(sorted(values), 0.5) if values else None


def quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between order statistics (the numpy default)"""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])

    position = q * (n - 1)
    lower = math.floor(position)
    upper = min(lower + 1, n - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def quantiles(values: Sequence[float], points: Sequence[float]) -> dict[str, float | None]:
    sorted_values = sorted(values)
    return {f"p{int(round(point * 100)):02d}": quantile(sorted_values, point) for point in points}


def histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Counts per fixed bin, the last bin includes its upper edge, values outside the edges are clipped"""
    counts = [0] * (len(edges) - 1)
    if not counts:
        return counts

    for value in values:
        counts[_bin_index(value, edges)] += 1

    return counts


def _bin_index(value: float, edges: Sequence[float]) -> int:
    last = len(edges) - 2
    if value >= edges[-1]:
        return last
    if value <= edges[0]:
        return 0

    for index in range(last + 1):
        if edges[index] <= value < edges[index + 1]:
            return index

    return last


def proportions(counts: Sequence[float]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(counts)
    return [count / total for count in counts]


def _smooth(dist: Sequence[float], eps: float) -> list[float]:
    smoothed = [max(value, eps) for value in dist]
    total = sum(smoothed)
    return [value / total for value in smoothed]


def psi(p: Sequence[float], q: Sequence[float], eps: float = 1e-4) -> float:
    """Population Stability Index between two distributions over the same bins"""
    if len(p) != len(q) or not p:
        return 0.0

    p_smooth, q_smooth = _smooth(p, eps), _smooth(q, eps)
    return sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p_smooth, q_smooth))


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-4) -> float:
    """Jensen-Shannon divergence in base 2, bounded in [0, 1]"""
    if len(p) != len(q) or not p:
        return 0.0

    p_smooth, q_smooth = _smooth(p, eps), _smooth(q, eps)
    mixture = [(pi + qi) / 2 for pi, qi in zip(p_smooth, q_smooth)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    return 0.5 * kl(p_smooth, mixture) + 0.5 * kl(q_smooth, mixture)


def chi2_contingency(table: Sequence[Sequence[float]]) -> tuple[float, float, int]:
    """Chi-square test of independence on a contingency table, returns (statistic, p-value, degrees of freedom)

    Rows or columns that are entirely zero are dropped before the test.
    """
    rows = [list(row) for row in table if sum(row) > 0]
    if not rows:
        return 0.0, 1.0, 0

    n_cols = len(rows[0])
    kept_columns = [c for c in range(n_cols) if sum(row[c] for row in rows) > 0]
    rows = [[row[c] for c in kept_columns] for row in rows]

    n_rows, n_cols = len(rows), len(kept_columns)
    if n_rows < 2 or n_cols < 2:
        return 0.0, 1.0, 0

    total = sum(sum(row) for row in rows)
    row_sums = [sum(row) for row in rows]
    col_sums = [sum(row[c] for row in rows) for c in range(n_cols)]

    stat = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = row_sums[r] * col_sums[c] / total
            if expected > 0:
                stat += (rows[r][c] - expected) ** 2 / expected

    dof = (n_rows - 1) * (n_cols - 1)
    return stat, chi2_survival(stat, dof), dof


def chi2_survival(stat: float, dof: int) -> float:
    """Upper tail probability of the chi-square distribution via the regularised incomplete gamma function"""
    if dof <= 0 or stat <= 0:
        return 1.0
    return _gammaincc(dof / 2.0, stat / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x), series for x < a + 1 and continued fraction otherwise"""
    if x < a + 1.0:
        return 1.0 - _gammainc_series(a, x)
    return _gammainc_continued_fraction(a, x)


def _gammainc_series(a: float, x: float, max_iter: int = 500, tol: float = 1e-14) -> float:
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(max_iter):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * tol:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_continued_fraction(a: float, x: float, max_iter: int = 500, tol: float = 1e-14) -> float:
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def histogram_quantile(counts: Sequence[int], edges: Sequence[float], q: float) -> float | None:
    """Quantile of a fixed-bin histogram by linear interpolation inside the bin that crosses it

    Assumes the values are spread uniformly inside each bin; the result is exact at the bin edges.
    """
    total = sum(counts)
    if total <= 0 or len(counts) != len(edges) - 1:
        return None

    target = q * total
    cumulative = 0.0
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        if cumulative + count >= target:
            fraction = (target - cumulative) / count
            return float(edges[index] + fraction * (edges[index + 1] - edges[index]))
        cumulative += count

    return float(edges[-1])
