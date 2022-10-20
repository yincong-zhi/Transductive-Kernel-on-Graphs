import tensorflow as tf
import gpflow
import gpflow.covariances as cov
from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingPoints

from .utils import matrix_polynomial, chebyshev_polynomial


class Chebyshev(gpflow.kernels.base.Kernel):
    def __init__(self, normalized_L, poly_degree, node_feats, base_kernel=None):
        super().__init__()
        self.num_nodes = normalized_L.shape[0]
        self.normalized_L = tf.cast(normalized_L, tf.float64) - tf.eye(self.num_nodes, dtype=tf.float64)
        self.coeffs = gpflow.Parameter(tf.ones(poly_degree+1))
        self.node_feats = tf.reshape(tf.range(self.num_nodes, dtype=normalized_L.dtype), (-1, 1)) if node_feats is None else tf.convert_to_tensor(node_feats)
        self.base_kernel = base_kernel

    def conv_mat(self):
        return chebyshev_polynomial(self.normalized_L, self.coeffs)

    def K(self, X, Y=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        if self.base_kernel is not None:
            cov = self.base_kernel.K(self.node_feats)
        else:
            cov = tf.eye(self.num_nodes, dtype=self.node_feats.dtype)
        conv_mat = self.conv_mat()
        cov = tf.matmul(conv_mat, tf.matmul(cov, tf.transpose(conv_mat)))
        return tf.gather(tf.gather(cov, X, axis=0), Y, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))


class SubgraphChebyshev(Chebyshev):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.normalized_L = None
            self.node_feats = None
            self.center_idcs = None
            self.chebyshev_mat = None

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
            self.normalized_L = normalized_L - tf.eye(normalized_L.shape[0],
                                                      dtype=normalized_L.dtype)
            self.chebyshev_mat = None

        def conv_mat(self):
            if self.chebyshev_mat is None:
                return super().conv_mat()
            return self.chebyshev_mat

        def K(self, X, Y=None):
            assert Y is None, "Unexpected argument Y"
            if self.base_kernel is not None:
                cov = self.base_kernel.K(X)
            else:
                cov = tf.eye(X, dtype=X.dtype)
            conv_mat = self.conv_mat()
            cov = tf.matmul(conv_mat, tf.matmul(cov, tf.transpose(conv_mat)))
            return cov

        def Kzx(self, Z, X):
            cov = self.base_kernel.K(Z, X)
            conv_mat = self.conv_mat()
            cov = tf.matmul(cov, tf.transpose(conv_mat))
            return cov

        def Kzz(self, Z, jitter=None):
            cov = self.base_kernel.K(Z)
            if jitter is not None:
                cov = cov + jitter * tf.eye(cov.shape[0], dtype=cov.dtype)
            return cov


@cov.Kuu.register(InducingPoints, SubgraphChebyshev)
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


@cov.Kuf.register(InducingPoints, SubgraphChebyshev, TensorLike)
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