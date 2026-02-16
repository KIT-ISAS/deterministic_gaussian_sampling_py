import deterministic_gaussian_sampling
import numpy as np
import pytest


def _compare_stats(cov_ref: np.ndarray,
                   dirac_pts: np.ndarray,
                   mean_ref=None,
                   rtol=5e-2,
                   atol=5e-2):
    """
    Compare empirical mean and covariance of Dirac points
    against reference mean and covariance.
    """

    cov_ref = np.asarray(cov_ref, dtype=float)
    dirac_pts = np.asarray(dirac_pts, dtype=float)

    assert dirac_pts.ndim == 2
    L, N = dirac_pts.shape

    if mean_ref is None:
        mean_ref = np.zeros(N, dtype=float)

    mean_emp = np.mean(dirac_pts, axis=0)
    cov_emp = np.cov(dirac_pts.T, ddof=0)

    np.testing.assert_allclose(
        mean_emp, mean_ref,
        rtol=rtol, atol=atol,
        err_msg="Mean mismatch"
    )

    np.testing.assert_allclose(
        cov_emp, cov_ref,
        rtol=rtol, atol=atol,
        err_msg="Covariance mismatch"
    )


# --------------------------------------------------
# Standard Normal Approximation
# --------------------------------------------------

@pytest.mark.parametrize("L,N", [
    (30, 2),
    (50, 3),
    (500, 3),
    (200, 9),
])
def test_snd_approximation(L, N, seed=42):

    snd_cov = np.eye(N, dtype=float)

    np.random.seed(seed)
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    result = g2dApprox.approximate_snd_double(L, N, x)

    _compare_stats(snd_cov, result.x)


# --------------------------------------------------
# Invalid Input Tests
# --------------------------------------------------

def test_snd_approximation_invalid_inputs():
    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()

    with pytest.raises(TypeError):
        g2dApprox.approximate_snd_double(5, 10, [[0.1]*10]*5)

    with pytest.raises(ValueError):
        g2dApprox.approximate_snd_double(5, 10, np.random.rand(5))

    with pytest.raises(ValueError):
        g2dApprox.approximate_snd_double(5, 10, np.random.rand(5, 10, 2))

    with pytest.raises(TypeError):
        g2dApprox.approximate_snd_double(
            5, 10,
            np.random.randint(0, 10, size=(5, 10))
        )

    with pytest.raises(ValueError):
        g2dApprox.approximate_snd_double(0, 10, np.random.rand(0, 10))

    with pytest.raises(ValueError):
        g2dApprox.approximate_snd_double(5, 0, np.random.rand(5, 0))

    with pytest.raises(ValueError):
        g2dApprox.approximate_snd_double(5, 10, np.random.rand(4, 10))


# --------------------------------------------------
# Diagonal Covariance Approximation
# --------------------------------------------------

@pytest.mark.parametrize("L,N", [
    (30, 2),
    (50, 3),
    (500, 3),
    (200, 9),
])
def test_full_covariance_approximation(L, N, seed=42):
    # create covariance matrix
    np.random.seed(seed)
    A = np.random.randn(N, N)
    cov = A @ A.T + 0.5 * np.eye(N)

    # random x values
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    result = g2dApprox.approximate_double(cov, L, N, x)

    _compare_stats(cov, result.x)

