import math
import pathlib
import numpy as np

import tensorflow as tf
import gpflow
from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import positive
import gpflow.covariances as cov

from .utils import matrix_polynomial, low, mexican_hat_normalized, get_approximation_projection_matrix

class Wavelet(gpflow.kernels.base.Kernel):
    def __init__(self, normalized_L, low_pass=1.0, scales=(1.0/5.0, 2.0/5.0), base_kernel=None,
                 node_feats=None, low_filter=low, band_filter=mexican_hat_normalized):
        super().__init__()
        self.num_scales = len(scales)
        self.base_kernel = base_kernel
        self.num_nodes = normalized_L.shape[0]
        self.node_feats = tf.reshape(tf.range(self.num_nodes, dtype=tf.float64), (-1, 1)) if node_feats is None else tf.convert_to_tensor(node_feats)
        # self.L = tf.cast(G.L.todense(), tf.float64)
        e, U = tf.linalg.eigh(normalized_L)

        self.e, self.U = tf.cast(e, tf.float64), tf.cast(U, tf.float64)
        self.low_f = low_filter
        if low_filter is not None:
            self.alpha = gpflow.Parameter(low_pass, transform=positive())
        self.band_f = band_filter
        for attr in range(1, self.num_scales + 1):
            self.__dict__['scale' + str(attr)] = gpflow.Parameter(scales[attr-1],
                                                                  transform=positive())

    def low(self):
        l = self.low_f(self.e, self.alpha, shift=0.0)
        return self.U @ tf.linalg.diag(l) @ tf.transpose(self.U)

    def band(self, scale=1.):
        # range = np.linspace(0, tf.reduce_max(self.e), 100)
        # range_band = range * scale * np.exp(-scale * range)
        l = self.band_f(self.e, scale, shift=0.0)
        return self.U @ tf.linalg.diag(l) @ tf.transpose(self.U)

    def wavelet(self):
        band = tf.math.reduce_sum([self.band(self.__dict__['scale' + str(attr)]) for attr in
                                   range(1, self.num_scales + 1)], axis=0)
        if self.low_f is not None:
            return band + self.low()
        return band

    def K(self, X, Y=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        if self.base_kernel is not None:
            cov = self.base_kernel.K(self.node_feats)
        else:
            cov = tf.eye(self.num_nodes, dtype=self.node_feats.dtype)
        t2 = self.wavelet() @ cov @ tf.transpose(self.wavelet())
        return tf.gather(tf.gather(t2, X), Y, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

class AdaptiveApproximateWavelet(gpflow.kernels.base.Kernel):
    def __init__(self, normalized_L, low_pass=1.0, scales=(1.0/5.0, 2.0/5.0), base_kernel=None,
                 poly_degree=5, node_feats=None, low_filter=low, band_filter=mexican_hat_normalized,
                 proj_mat_path=None, sd_steps_divider=64, sd_degree=10, sd_samples=100, seed=1):
        """
        :param normalized_L: The normalized graph Laplacian that has not yet been shifted, i.e. the
        eigenvalues are between [0, 2].
        """
        super().__init__()
        self.num_scales = len(scales)
        self.base_kernel = base_kernel
        self.num_nodes = normalized_L.shape[0]
        self.node_feats = tf.reshape(tf.range(self.num_nodes, dtype=tf.float64), (-1, 1)) if node_feats is None else tf.convert_to_tensor(node_feats)
        self.normalized_L = tf.cast(normalized_L, tf.float64) - tf.eye(self.num_nodes, dtype=tf.float64)
        self.num_approx_points = len(self.normalized_L)//8
        self.approx_points = tf.convert_to_tensor(np.linspace(-1.0, 1.0, self.num_approx_points), dtype=self.normalized_L.dtype)
        self.low_f = low_filter
        self.band_f = band_filter

        # Load pre-computed or recompute projection matrix
        proj_mat_path = "proj_mat.npy" if proj_mat_path is None else proj_mat_path
        pre_comp_file = pathlib.Path(proj_mat_path)
        if pre_comp_file.exists():
            proj_mat = np.load(str(pre_comp_file))
            assert proj_mat.shape[0] == poly_degree+1 and proj_mat.shape[1] == self.num_approx_points, "Pre-computed projection matrix does not match specified parameters"
        else:
            proj_mat = get_approximation_projection_matrix(
                self.normalized_L.numpy(), poly_degree, self.num_approx_points,
                sd_steps=len(self.normalized_L) // sd_steps_divider, sd_degree=sd_degree,
                sd_samples=sd_samples, plot=False)
            np.save(str(pre_comp_file), proj_mat)
        self.proj_mat = tf.convert_to_tensor(proj_mat, dtype=self.normalized_L.dtype)

        # Kernel parameters
        if low_filter is not None:
            self.alpha = gpflow.Parameter(low_pass, transform=positive())
        for attr in range(1, self.num_scales + 1):
            self.__dict__['scale' + str(attr)] = gpflow.Parameter(scales[attr-1], transform=positive())

    def low(self):
        y = self.low_f(self.approx_points, alpha=self.alpha)
        y_hat = tf.linalg.matvec(self.proj_mat, y)
        filter = matrix_polynomial(self.normalized_L, y_hat)
        return filter

    def band(self, scale=1., idx=0):
        y = self.band_f(self.approx_points, scale)
        y_hat = tf.linalg.matvec(self.proj_mat, y)
        filter = matrix_polynomial(self.normalized_L, y_hat)
        return filter

    def wavelet(self):
        band = tf.math.reduce_sum([self.band(self.__dict__['scale' + str(attr)], attr-1) for attr in
                                   range(1, self.num_scales + 1)], axis=0)
        if self.low_f is not None:
            low = self.low()
            return tf.cast(band, low.dtype) + low
        return band

    # @tf.function
    def K(self, X, Y=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        if self.base_kernel is not None:
            cov = self.base_kernel.K(self.node_feats)
        else:
            cov = tf.eye(self.num_nodes, dtype=self.node_feats.dtype)
        wavelet_filter = self.wavelet()
        t2 = tf.matmul(wavelet_filter, tf.matmul(cov, tf.transpose(wavelet_filter)))
        return tf.gather(tf.gather(t2, X), Y, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

    def Kzx(self, Z, X):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        cov = self.base_kernel.K(Z, self.node_feats)
        wavelet_filter = self.wavelet()
        cov = tf.matmul(cov, tf.transpose(wavelet_filter))
        return tf.gather(cov, X, axis=1)

    def Kzz(self, Z, jitter=None):
        cov = self.base_kernel.K(Z)
        if jitter is not None:
            cov = cov + jitter * tf.eye(cov.shape[0], dtype=cov.dtype)
        return cov


class SubgraphAdaptiveApproximateWavelet(AdaptiveApproximateWavelet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normalized_L = None
        self.node_feats = None
        self.center_idcs = None
        self.wavelet_filter = None

    def set_subgraph(self, normalized_L, center_idcs):
        """
        A convolution operation may be applied only for the subgraph relevant to the current batch.
        This saves a lot of time as it reduces the size of the convolution and covariance matrix
        from the number of all nodes to the number of nodes in the image of the convolution
        operation.
        :param conv_mat: Convolution matrix of shape [subgraph_size, subgraph_size]. May either be
        a sparse or dense tensor.
        :param center_idcs: The subgraph contains nodes that are required for performing the
        convolution but which may not be part of the minibatch for which we would like to compute
        the covariance. Therefore, this Tensor contains the indices of the nodes in the subgraph,
        for which we want to compute the covariance. The indexing is relative to the subgraph.
        For example let's say our graph has 10 nodes in total, our minibatch contains nodes 4 and 8
        and nodes 2, 3, 4, 6, 8 are required to compute the convolution (i.e. these nodes are in the
        domain of the convolution operation). Then these nodes form the subgraph with a 5x5
        convolution matrix and center_idcs is equal to [2, 5].
        """
        self.center_idcs = center_idcs
        normalized_L = tf.sparse.to_dense(tf.sparse.reorder(normalized_L))
        self.normalized_L = normalized_L - tf.eye(normalized_L.shape[0], dtype=normalized_L.dtype)
        self.wavelet_filter = None

    def wavelet(self):
        if self.wavelet_filter is None:
            return super().wavelet()
        return self.wavelet_filter

    def K(self, X, Y=None):
        assert Y is None, "Unexpected argument Y"
        cov = self.base_kernel.K(X)
        wavelet_filter = self.wavelet()
        cov = tf.matmul(wavelet_filter, tf.matmul(cov, wavelet_filter, adjoint_b=True))
        # return tf.gather(tf.gather(t2, X), Y, axis=1)
        return cov

    def Kzx(self, Z, X):
        cov = self.base_kernel.K(Z, X)
        wavelet_filter = self.wavelet()
        cov = tf.matmul(cov, tf.transpose(wavelet_filter))
        return cov


@cov.Kuu.register(InducingPoints, AdaptiveApproximateWavelet)
def Kuu_wavelet_adaptive(inducing_variable, kernel, jitter=None):
    """
    Computes the covariance matrix between the inducing points (which are not
    associated with any node).
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :return: Covariance matrix between the inducing variables.
    """
    return kernel.Kzz(inducing_variable.Z, jitter=jitter)


@cov.Kuf.register(InducingPoints, AdaptiveApproximateWavelet, TensorLike)
def Kuf_wavelet_adaptive(inducing_variable, kernel, X):
    """
    Computes the covariance matrix between inducing points (which are not
    associated with any node) and normal inputs.
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :param X: Normal inputs. Note, however, that to simplify the
    implementation, we pass in the indices of the nodes rather than their
    features directly.
    :return: Covariance matrix between inducing variables and inputs.
    """
    return kernel.Kzx(inducing_variable.Z, X)


@cov.Kuu.register(InducingPoints, SubgraphAdaptiveApproximateWavelet)
def Kuu_wavelet_adaptive(inducing_variable, kernel, jitter=None):
    """
    Computes the covariance matrix between the inducing points (which are not
    associated with any node).
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :return: Covariance matrix between the inducing variables.
    """
    return kernel.Kzz(inducing_variable.Z, jitter=jitter)


@cov.Kuf.register(InducingPoints, SubgraphAdaptiveApproximateWavelet, TensorLike)
def Kuf_wavelet_adaptive(inducing_variable, kernel, X):
    """
    Computes the covariance matrix between inducing points (which are not
    associated with any node) and normal inputs.
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :param X: Normal inputs. Note, however, that to simplify the
    implementation, we pass in the indices of the nodes rather than their
    features directly.
    :return: Covariance matrix between inducing variables and inputs.
    """
    return kernel.Kzx(inducing_variable.Z, X)