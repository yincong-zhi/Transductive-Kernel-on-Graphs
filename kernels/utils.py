import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def low_pass_filter_from_string(name):
    if name == "low":
        return low
    elif name == "none":
        return None
    else:
        raise NotImplementedError(f"No low pass filter named {name} found.")


def band_pass_filter_from_string(name):
    if name == "mexican_hat":
        return mexican_hat
    elif name == "mexican_hat_normalized":
        return mexican_hat_normalized
    elif name == "morlet":
        return morlet
    elif name == "scaled_morlet":
        return scaled_morlet
    else:
        raise NotImplementedError(f"No band pass filter named {name} found.")


def low(x, alpha=1., shift=1.0):
    x = x + shift
    return 1. / (1. + alpha * x)


def mexican_hat(x, scale=0.3, shift=1.0):
    x = x + shift
    if isinstance(x, np.ndarray):
        return x * scale * np.exp(-scale * x)
    return x * scale * tf.exp(-scale * x)


def mexican_hat_normalized(x, scale=0.3, shift=1.0):
    x = x + shift
    x = x * scale
    if isinstance(x, np.ndarray):
        y = 2.0 * np.sqrt(2.0 / 3.0) * np.power(np.pi, -1.0/4.0) * x**2 * np.exp(-0.5 * x**2)
        # y = x**2 * np.exp(-x**2)
    else:
        pi = tf.convert_to_tensor(np.pi, dtype=x.dtype)
        const = tf.cast(2.0 * tf.sqrt(2.0 / 3.0), dtype=x.dtype)
        const *= tf.cast(tf.pow(pi, -1.0 / 4.0), dtype=x.dtype)
        y = const * x**2 * tf.exp(-0.5 * x ** 2)
        # y = x**2 * tf.exp(-x**2)
    return y


def mexican_hat_normalized2(x, scale=0.3, shift=1.0):
    x = x + shift
    if isinstance(x, np.ndarray):
        y = np.sqrt(8) * np.power(scale, 5.0/2.0) * np.power(np.pi, 1.0/4.0) / np.sqrt(3) * x**2 * np.exp(-0.5 * scale**2 * x**2)
        # y = 2.0 * np.sqrt(2.0 / 3.0) * np.power(np.pi, -1.0/4.0) * x**2 * np.exp(-0.5 * x**2)
        # y = x**2 * np.exp(-x**2)
    else:
        pi = tf.convert_to_tensor(np.pi, dtype=x.dtype)
        scale = tf.convert_to_tensor(scale, dtype=x.dtype)
        const = tf.cast(tf.sqrt(8.0 / 3.0), dtype=x.dtype)
        const *= tf.cast(tf.pow(pi, 1.0 / 4.0), dtype=x.dtype)
        const *= tf.cast(tf.pow(scale, 5.0/2.0), dtype=x.dtype)
        y = const * x**2 * tf.exp(-0.5 * scale**2 * x**2)
        # y = x**2 * tf.exp(-x**2)
    return y


def morlet(x, sigma=1., shift=1.0):
    x = x + shift
    exp_f = np.exp if isinstance(x, np.ndarray) else tf.exp
    pi = np.pi if isinstance(x, np.ndarray) else tf.cast(np.pi, x.dtype)
    c_sigma = (1. + exp_f(-sigma**2.) - 2. * exp_f(-3./4. * sigma**2)) ** (-1./2.)
    kappa_sigma = exp_f(-1./2. * sigma**2)
    wavelet = c_sigma * (pi ** (-1./4.)) * (exp_f(-1./2. * (sigma - x)**2) - kappa_sigma * exp_f(-1./2. * x**2))
    return wavelet


def scaled_morlet(x, sigma=1., scaling=10.0, shift=1.):
    x = x + shift
    c_sigma = (1. + tf.math.exp(-sigma ** 2.) - 2. * tf.math.exp(-3. / 4. * sigma ** 2)) ** (
                -1. / 2.)
    kappa_sigma = tf.math.exp(-1. / 2. * sigma ** 2)
    pi4 = tf.cast(np.pi ** (-1. / 4.), tf.float64)
    wavelet_fourier = c_sigma * pi4 * (tf.math.exp(-1. / 2. * (sigma - (x * scaling)) ** 2) - (
                kappa_sigma * tf.math.exp(-1. / 2. * (x * scaling) ** 2)))
    return wavelet_fourier


def compute_full_filter(x, low_scale, band_scales, low_filter, band_filter):
    full = low_filter(x, low_scale)
    for idx in range(len(band_scales)):
        full += band_filter(x, band_scales[idx])
    return full


def matrix_polynomial(mat, coefficients):
    polynomial = None
    monomial = tf.eye(mat.shape[0], dtype=mat.dtype)
    for coeff in coefficients:
        polynomial = polynomial + coeff * monomial if polynomial is not None else coeff * monomial
        if isinstance(mat, tf.sparse.SparseTensor):
            monomial = tf.transpose(tf.sparse.sparse_dense_matmul(mat, monomial), [1, 0])
        else:
            monomial = tf.matmul(monomial, mat)
    return polynomial


def chebyshev_polynomial(mat, coefficients):
    monomial_prev_exists = False        # Using these helper variables is required to make it work with the SciPy optimizer
    monomial_exists = False
    monomial_prev = tf.zeros_like(mat)
    monomial = tf.zeros_like(mat)
    polynomial = None
    for coeff in coefficients:
        if not monomial_exists:
            monomial = tf.eye(mat.shape[0], dtype=mat.dtype)
            monomial_exists = True
        elif not monomial_prev_exists:
            monomial_prev = monomial
            monomial = mat
            monomial_prev_exists = True
        else:
            temp = 2 * mat @ monomial - monomial_prev
            monomial_prev = monomial
            monomial = temp
        polynomial = coeff * monomial if polynomial is None else polynomial + coeff * monomial
    return polynomial


def scalar_polynomial(x, coefficients):
    polynomial = None
    monomial = np.ones_like(x)
    for coeff in coefficients:
        polynomial = polynomial + coeff * monomial if polynomial is not None else coeff * monomial
        monomial *= x
    return polynomial

def gamma(j, a, b):
    if j == 0:
        return 1.0 / np.pi * (np.arccos(a) - np.arccos(b))
    return 2.0 / np.pi * ((np.sin(j * np.arccos(a)) - np.sin(j * np.arccos(b))) / j)
    
def g(j, p):
    alpha_p = np.pi / (p + 2.0)
    g_j_p = ((1.0 - (j / (p + 2))) * np.sin(alpha_p) * np.cos(j * alpha_p) + 1.0 / (p + 2) * np.cos(alpha_p) * np.sin(j * alpha_p)) / np.sin(alpha_p)
    return g_j_p

def recursive_w(matrix, vec, w_j, w_jj):
    if w_j is None:
        return vec              # j = 0
    if w_jj is None:
        return matrix @ vec     # j = 1
    return 2.0 * matrix @ w_j - w_jj

def estimate_number_eigenvals(matrix, degree, num_samples, a, b):
    vals = []
    for _ in range(num_samples):
        vec = np.random.normal(0.0, 1.0, size=len(matrix))
        w_j = None
        w_jj = None
        val = 0.0
        for j in range(degree):
            # Compute recursive w
            w = recursive_w(matrix, vec, w_j, w_jj)
            w_jj = w_j
            w_j = w
            # Compute value for current degree
            g_j_p = g(j, degree)
            gamma_j = gamma(j, a, b)
            val += g_j_p * gamma_j * vec.T @ w
        vals.append(val)
    return np.mean(vals)

def estimate_spectral_density(matrix, num_steps, degree, num_samples, plot=False):
    steps = np.linspace(-1.0, 1.0, num_steps)
    # print(steps)
    mus = []
    for b in steps:
        mu = estimate_number_eigenvals(matrix, degree, num_samples, -1.0, b)
        mus.append(mu)
        print(b, mu)

    inter = PchipInterpolator(steps, np.array(mus))
    spectral_density = inter.derivative()

    if plot:
        plt.xlabel("eigenvalue")
        plt.ylabel("cumulative spectral density")
        plt.title("Estimated cumulative spectral density")
        plt.plot(steps, inter(steps))
        plt.show()
    return spectral_density
    
def get_approximation_projection_matrix(matrix, degree, num_steps, sd_steps, sd_degree, sd_samples,
                                        plot=False):
    """
    We aim to compute the polynomial approximation of the filter function at linearly spaced
    points on the real line with higher weights where there are more eigenvalues on that
    line. This can be done by projecting the unapproximated filter function values y at the
    linearly spaced points using the weighted projection matrix to get polynomial coefficients
        c = [V^T W V]^-1 V^T y = P y,
    where V is the Vandermonde matrix, i.e.
        [[1     x1      x1^2        x1^3        ...]
         [1     x2      x2^2        x2^3        ...]
         [1     x3      x3^2        x3^3        ...]]
    and W is a diagonal matrix with the weights of the points on the real line.
    """
    ls = np.linspace(-1.0, 1.0, num_steps)
    V = np.vander(ls, N=degree+1, increasing=True)
    spectral_density = estimate_spectral_density(matrix, sd_steps, sd_degree, sd_samples, plot)
    W = spectral_density(ls)
    W = np.diag(W / np.sum(W))
    P = np.linalg.inv(V.T @ W @ V) @ V.T @ W
    return P

if __name__ == '__main__':
    x = np.linspace(0.0, 2.0, num=100)
    y = mexican_hat_normalized(x, scale=0.8)
    plt.plot(x, y)
    plt.show()