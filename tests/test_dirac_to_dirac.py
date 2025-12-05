import deterministic_gaussian_sampling
import numpy as np
from scipy import stats
import pytest

def compare_stats(full: np.ndarray, reduced: np.ndarray, rtol=1e-1, atol=1e-2):
    """
    Compare the statistics of two datasets.
    Raises AssertionError if they differ more than allowed tolerances.
    """
    assert full.ndim == 2 and reduced.ndim == 2
    assert full.shape[1] == reduced.shape[1]  # same dimensionality

    # Compute mean, variance, std
    stats_full = {
        'mean': full.mean(axis=0),
        'var': full.var(axis=0, ddof=1),
        'std': full.std(axis=0, ddof=1)
    }
    stats_red = {
        'mean': reduced.mean(axis=0),
        'var': reduced.var(axis=0, ddof=1),
        'std': reduced.std(axis=0, ddof=1)
    }

    # Assert that means and stds are approximately equal
    np.testing.assert_allclose(stats_full['mean'], stats_red['mean'], rtol=rtol, atol=atol)
    np.testing.assert_allclose(stats_full['std'], stats_red['std'], rtol=rtol, atol=atol)

    # Optionally: per-dimension Welch's t-test
    for dim in range(full.shape[1]):
        t_stat, p_val = stats.ttest_ind(full[:, dim], reduced[:, dim], equal_var=False)
        # Expect p-value > 0.05 (no significant difference)
        assert p_val > 0.05, f"Significant difference in dimension {dim}: p={p_val:.3f}"


@pytest.mark.parametrize("M,L,N", [
    (100, 3, 2),
    (100, 5, 8),
    (250, 5, 3),
    (250, 20, 9),
    (500, 5, 3),
    # (500, 75, 20),
    # (750, 5, 3),
    # (750, 50, 12),
    # (1000, 100, 3),
    # (1000, 100, 15),
])
def test_dirac_approximation_simple(M, L, N, seed=42):
    np.random.seed(seed)
    x = np.random.rand(L, N)
    y = np.random.rand(M, N)
    d2dApprox = deterministic_gaussian_sampling.DiracToDiracApproximation()
    result = d2dApprox.approximate_double(y, M, L, N, x)

    # Compare statistics between original and reduced sets
    compare_stats(x, result.x)

@pytest.mark.parametrize("M,L,N", [
    (100, 3, 2),
    (100, 5, 8),
    (250, 5, 3),
    (250, 20, 9),
    (500, 5, 3),
    # (500, 75, 20),
    # (750, 5, 3),
    # (750, 50, 12),
    # (1000, 100, 3),
    # (1000, 100, 15),
])
def test_dirac_approximation_threaded(M, L, N, seed=42):
    np.random.seed(seed)
    x = np.random.rand(L, N)
    y = np.random.rand(M, N)
    d2dApprox = deterministic_gaussian_sampling.DiracToDiracApproximation()
    result = d2dApprox.approximate_thread_double(y, M, L, N, x)

    # Compare statistics between original and reduced sets
    compare_stats(x, result.x)
