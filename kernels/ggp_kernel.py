import tensorflow as tf
import gpflow


class GraphGP(gpflow.kernels.base.Kernel):
    def __init__(self, adj_mat, base_kernel=None, node_feats=None, seed=1):
        """
        :param normalized_L: The normalized graph Laplacian that has not yet been shifted, i.e. the
        eigenvalues are between [0, 2].
        """
        super().__init__()
        self.base_kernel = base_kernel
        self.num_nodes = adj_mat.shape[0]
        self.node_feats = tf.reshape(tf.range(self.num_nodes, dtype=tf.float64), (-1, 1)) if node_feats is None else tf.convert_to_tensor(node_feats, dtype=tf.float64)
        # self.normalized_L = tf.cast(normalized_L, tf.float64) - tf.eye(self.num_nodes, dtype=tf.float64)
        self.P = tf.cast(adj_mat / adj_mat.sum(axis=1), tf.float64)

    # @tf.function
    def K(self, X, Y=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X
        if self.base_kernel is not None:
            cov = self.base_kernel.K(self.node_feats)
        else:
            cov = tf.eye(self.num_nodes, dtype=self.node_feats.dtype)
        t2 = tf.matmul(self.P, tf.matmul(cov, tf.transpose(self.P)))
        return tf.gather(tf.gather(t2, X), Y, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

    def Kzx(self, Z, X):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        cov = self.base_kernel.K(Z, self.node_feats)
        cov = tf.matmul(cov, tf.transpose(self.P))
        return tf.gather(cov, X, axis=1)

    def Kzz(self, Z, jitter=None):
        cov = self.base_kernel.K(Z)
        if jitter is not None:
            cov = cov + jitter * tf.eye(cov.shape[0], dtype=cov.dtype)
        return cov
