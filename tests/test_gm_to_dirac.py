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


# --------------------------------------------------
# Distance / Gradient Evaluation
# --------------------------------------------------

def test_snd_distance_and_gradient(seed=42):
    L, N = 30, 2

    np.random.seed(seed)
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    snd = g2dApprox.approximate_snd_double
    r = snd(L, N, x)

    d_opt = snd.modified_van_mises_distance_sq(L, N, r.x)
    d_rand = snd.modified_van_mises_distance_sq(L, N, np.random.randn(L, N))
    assert np.isfinite(d_opt)
    assert np.isfinite(d_rand)
    assert d_opt < d_rand

    d_opt_50 = snd.modified_van_mises_distance_sq(L, N, r.x, bMax=50)
    assert np.isfinite(d_opt_50)
    assert d_opt_50 != d_opt

    grad = snd.modified_van_mises_distance_sq_derivative(L, N, r.x)
    assert grad.shape == (L, N)
    assert np.all(np.isfinite(grad))


def test_full_covariance_distance_and_gradient(seed=42):
    L, N = 30, 3

    # create covariance matrix
    np.random.seed(seed)
    A = np.random.randn(N, N)
    cov = A @ A.T + 0.5 * np.eye(N)

    # random x values
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    full = g2dApprox.approximate_double
    r = full(cov, L, N, x)

    d = full.modified_van_mises_distance_sq(cov, L, N, r.x)
    assert np.isfinite(d)

    grad = full.modified_van_mises_distance_sq_derivative(cov, L, N, r.x)
    assert grad.shape == (L, N)
    assert np.all(np.isfinite(grad))


# --------------------------------------------------
# Even Closed-Form Approximation
# --------------------------------------------------

@pytest.mark.parametrize("L,N", [
    (40, 2),
    (100, 4),
])
def test_even_closed_form_approximation(L, N, seed=42, bMax=100.0):

    snd_cov = np.eye(N, dtype=float)

    np.random.seed(seed)
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    even = g2dApprox.approximate_even_double
    r = even(L, N, bMax, x)

    _compare_stats(snd_cov, r.x)

    x_rand = np.random.randn(L, N)
    d_opt = even.modified_van_mises_distance_sq(L, N, bMax, r.x)
    d_rand = even.modified_van_mises_distance_sq(L, N, bMax, x_rand)
    assert d_opt < d_rand

    grad = even.modified_van_mises_distance_sq_derivative(L, N, bMax, x_rand)
    assert grad.shape == (L, N)
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


# --------------------------------------------------
# Even bMax-Free Approximation
# --------------------------------------------------

@pytest.mark.parametrize("L,N", [
    (40, 2),
    (100, 4),
])
def test_even_binf_approximation(L, N, seed=42):

    snd_cov = np.eye(N, dtype=float)

    np.random.seed(seed)
    x = np.random.rand(L, N).astype(np.float64)

    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    binf = g2dApprox.approximate_even_binf_double
    r = binf(L, N, x)

    _compare_stats(snd_cov, r.x)

    x_rand = np.random.randn(L, N)
    d_opt = binf.modified_van_mises_distance_sq(L, N, r.x)
    d_rand = binf.modified_van_mises_distance_sq(L, N, x_rand)
    assert d_opt < d_rand

    grad = binf.modified_van_mises_distance_sq_derivative(L, N, x_rand)
    assert grad.shape == (L, N)
    assert np.all(np.isfinite(grad))


# --------------------------------------------------
# Even Invalid Input Tests
# --------------------------------------------------

def test_even_approximation_invalid_inputs():
    g2dApprox = deterministic_gaussian_sampling.GaussianToDiracApproximation()
    even = g2dApprox.approximate_even_double
    binf = g2dApprox.approximate_even_binf_double
    L = 10

    # odd dimension
    x_odd = np.random.rand(L, 3)
    with pytest.raises(ValueError):
        even(L, 3, 100.0, x_odd)
    with pytest.raises(ValueError):
        even.approximate_isotropic(L, 3, 1.0, 100.0, x_odd)
    with pytest.raises(ValueError):
        even.modified_van_mises_distance_sq(L, 3, 100.0, x_odd)
    with pytest.raises(ValueError):
        even.modified_van_mises_distance_sq_derivative(L, 3, 100.0, x_odd)
    with pytest.raises(ValueError):
        even.modified_van_mises_distance_sq_isotropic(L, 3, 1.0, 100.0, x_odd)
    with pytest.raises(ValueError):
        binf(L, 3, x_odd)
    with pytest.raises(ValueError):
        binf.modified_van_mises_distance_sq(L, 3, x_odd)
    with pytest.raises(ValueError):
        binf.modified_van_mises_distance_sq_derivative(L, 3, x_odd)

    # non-positive bMax
    x = np.random.rand(L, 2)
    for bMax in (0.0, -1.0):
        with pytest.raises(ValueError):
            even(L, 2, bMax, x)
        with pytest.raises(ValueError):
            even.modified_van_mises_distance_sq(L, 2, bMax, x)
        with pytest.raises(ValueError):
            even.modified_van_mises_distance_sq_derivative(L, 2, bMax, x)

    # non-positive sigma
    with pytest.raises(ValueError):
        even.approximate_isotropic(L, 2, 0.0, 100.0, x)
    with pytest.raises(ValueError):
        even.modified_van_mises_distance_sq_isotropic(L, 2, 0.0, 100.0, x)
