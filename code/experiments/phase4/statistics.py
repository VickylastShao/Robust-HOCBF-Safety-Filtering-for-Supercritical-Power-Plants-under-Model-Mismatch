"""Statistical analysis tools for Phase 4 experiment results.

Provides confidence intervals, hypothesis tests, and bootstrap methods
for rigorously evaluating safety violation rates and performance metrics.
"""
import numpy as np
from scipy import stats as sp_stats


def wilson_ci(count: int | float, n: int | float, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    count : number of successes (violations)
    n : total number of trials (steps)
    alpha : significance level (default 0.05 for 95% CI)

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = count / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return (float(p_hat), float(ci_lower), float(ci_upper))


def permutation_test(a: list | np.ndarray, b: list | np.ndarray,
                     n_perm: int = 10000, seed: int | None = None) -> tuple[float, float]:
    """Two-sided permutation test for difference in means.

    Tests H0: mean(a) = mean(b) vs H1: mean(a) != mean(b).

    Parameters
    ----------
    a, b : arrays of observations (e.g., per-seed violation rates)
    n_perm : number of permutations
    seed : random seed for reproducibility

    Returns
    -------
    (observed_diff, p_value)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed_diff = float(np.mean(a) - np.mean(b))

    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    count_extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)
    return (observed_diff, float(p_value))


def bootstrap_ci(data: list | np.ndarray, stat_fn=None, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int | None = None) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an arbitrary statistic.

    Parameters
    ----------
    data : array of observations
    stat_fn : function applied to each bootstrap sample (default: np.mean)
    n_boot : number of bootstrap resamples
    alpha : significance level
    seed : random seed

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    data = np.asarray(data, dtype=float)
    if stat_fn is None:
        stat_fn = np.mean
    point_est = float(stat_fn(data))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (point_est, ci_lower, ci_upper)


def format_violation_with_ci(count: int, n: int, alpha: float = 0.05) -> str:
    """Format violation rate with Wilson CI for LaTeX tables.

    Returns a string like "0.12 (0.08, 0.16)" or "<0.02 [0, 0.02]".
    """
    rate, ci_lo, ci_hi = wilson_ci(count, n, alpha)
    if rate < 0.005 and ci_lo < 0.001:
        return f"$<${ci_hi * 100:.2f}\\%"
    pct = rate * 100
    lo_pct = ci_lo * 100
    hi_pct = ci_hi * 100
    return f"{pct:.2f}\\% ({lo_pct:.2f}, {hi_pct:.2f})"


def aggregate_seeds_to_ci(seed_results: list[dict], key: str = 'cbf_violation_rate') -> dict:
    """Aggregate per-seed results into overall stats with CI.

    Parameters
    ----------
    seed_results : list of result dicts, each with (mean, std) tuples
    key : metric key to aggregate

    Returns
    -------
    dict with 'mean', 'std', 'wilson_ci', 'bootstrap_ci', 'n_seeds'
    """
    values = []
    for r in seed_results:
        v = r.get(key)
        if v is not None:
            if isinstance(v, (list, tuple)):
                values.append(v[0])
            else:
                values.append(float(v))

    if not values:
        return {'mean': 0.0, 'std': 0.0, 'n_seeds': 0}

    values = np.array(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    bs_ci = bootstrap_ci(values)

    return {
        'mean': mean_val,
        'std': std_val,
        'bootstrap_ci': bs_ci,
        'n_seeds': len(values),
    }
