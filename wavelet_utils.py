import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def plot_spectral_filters(low_scale, band_scales, low_filter, band_filter, eigvals, pred_low_scale=None,
                          pred_band_scales=None, approx=False, approx_low_filter=None, approx_band_filter=None,
                          eigvals_bottom=False, save_pdf=False, figsize=None):
    N = 4*len(eigvals)
    x = np.linspace(eigvals.min(), eigvals.max(), N)

    fig, ax = plt.subplots(figsize=figsize)

    if eigvals_bottom:
        ax.plot(eigvals, np.zeros_like(eigvals), 'x', markersize=8.0, label="eigenvalues")

    # Plot low pass filter
    ax.plot(x, low_filter(x, low_scale), color="blue", linewidth=3.0, label="low pass")
    if not eigvals_bottom:
        ax.plot(eigvals, low_filter(eigvals, low_scale), 'x', color="blue")
    if pred_low_scale is not None:
        if approx:
            ax.plot(x, approx_low_filter(x, pred_low_scale), color="blue", linewidth=3.0, linestyle="dashed", label="predicted low pass (approx)")
            ax.plot(x, low_filter(x, pred_low_scale), color="blue", linewidth=3.0, linestyle="dotted", label="predicted low pass", alpha=0.5)
        else:
            ax.plot(x, low_filter(x, pred_low_scale), color="blue", linewidth=3.0, linestyle="dashed", label="predicted low pass")

    # Plot band pass filters
    colors = ["red", "orange", "yellow", "green"]
    for idx in range(len(band_scales)):
        ax.plot(x, band_filter(x, band_scales[idx]), color=colors[idx], linewidth=3.0, label=f"band pass {idx}")
        if not eigvals_bottom:
            ax.plot(eigvals, band_filter(eigvals, band_scales[idx]), 'x', color=colors[idx])
        if pred_band_scales is not None:
            if approx:
                ax.plot(x, approx_band_filter(x, pred_band_scales[idx]), color=colors[idx],
                         linewidth=3.0, linestyle="dashed", label=f"predicted band pass {idx} (approx)")
                ax.plot(x, band_filter(x, pred_band_scales[idx]), color=colors[idx], linewidth=3.0, linestyle="dotted", label="predicted band pass",
                         alpha=0.5)
            else:
                ax.plot(x, band_filter(x, pred_band_scales[idx]), color=colors[idx],
                         linestyle="dashed", linewidth=3.0, label=f"predicted band pass {idx}")

    # Plot complete filter
    full = compute_full_filter(x, low_scale, band_scales, low_filter, band_filter)
    full_eigvals = compute_full_filter(eigvals, low_scale, band_scales, low_filter, band_filter)
    if pred_low_scale is not None:
        if approx:
            pred_full = compute_full_filter(x, pred_low_scale, pred_band_scales, approx_low_filter, approx_band_filter)
        else:
            pred_full = compute_full_filter(x, pred_low_scale, pred_band_scales, low_filter, band_filter)
        plt.plot(x, pred_full, color="black", linestyle="dashed", linewidth=3.0, label="predicted full filter (approx)")
    ax.plot(x, full, color="black", linewidth=3.0, label="full filter")
    if not eigvals_bottom:
        ax.plot(eigvals, full_eigvals, 'x', color="black")

    title = f"low pass={low_scale:.2f}"
    if pred_low_scale is not None:
        title += f" ({pred_low_scale:.2f})"
    for idx in range(len(band_scales)):
        title += f" band pass {idx}={band_scales[idx]:.2f}"
        if pred_band_scales is not None:
            title += f" ({pred_band_scales[idx]:.2f})"
    ax.set_title(title)

    ax.legend(prop={'size': 11}, loc=(0.75, 0.15))
    if save_pdf:
        fig.savefig("filters.pdf")
    fig.show()


if __name__ == '__main__':
    x = np.linspace(0.0, 2.0, num=100)
    y = mexican_hat_normalized(x, scale=0.8)
    plt.plot(x, y)
    plt.show()