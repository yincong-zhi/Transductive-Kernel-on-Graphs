import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_swiss_roll, make_s_curve
import gpflow
from gpflow.utilities import print_summary, set_trainable, positive

import matplotlib
font = {'size'   : 15}
matplotlib.rc('font', **font)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="transductive", type=str, help='gp, graph_only, transductive, ggp, wavelet')
parser.add_argument("--seed", default=3, type=int, help='integer for generating the graph')
parser.add_argument("--plot", default=False, type=bool, help='plot ground truth signal')
parser.add_argument("--training", default=10, type=int, help='number of training points')
parser = parser.parse_args()
model = parser.model

coord, labels = make_swiss_roll(1000, random_state = parser.seed)
#labels += np.random.randn(1000)
#coord, labels = make_s_curve(1000)
#coord = (coord - np.mean(coord))/np.std(coord)

labels -= np.mean(labels)
#labels /= np.std(labels)

from pygsp import graphs
import networkx as nx
G = graphs.NNGraph(coord, k = 4)
#print('is graph connected:', G.is_connected())
G.compute_laplacian('normalized')
G.set_coordinates(coord)

training = parser.training
data, t = coord[:training], labels[:training]
test_data, test_t = coord[training:], labels[training:]

# plotting
if parser.plot:
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=test_t, marker='o', s=10)
    p2 = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c='red', s=50, marker='o', clim = [labels.min(), labels.max()])
    plt.colorbar(p)
    plt.title('training (red) and test points')
    plt.show()

    G.plot_signal(labels, vertex_size = 10)
    plt.title('Ground Truth')
    plt.show()

def plot(pred, MAE = None):
    G.plot_signal(tf.concat((t, tf.reshape(pred, -1)), axis = 0).numpy(), vertex_size = 10)
    plt.title('Prediction, MAE = {:.2f}'.format(MAE))
    plt.show()

class graph_diff(gpflow.kernels.Kernel):
    def __init__(self, g_sigma = 1., g_var = 1.):
        super().__init__(active_dims=None)
        self.g_sigma = gpflow.Parameter(g_sigma, transform = positive())
        self.g_var = gpflow.Parameter(g_var, transform = positive())
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        #self.sparse_L = tf.cast(convert_sparse_matrix_to_sparse_tensor(nx.laplacian_matrix(G)), tf.float64)
        self.sparse_L = tf.cast(convert_sparse_matrix_to_sparse_tensor(G.L), tf.float64)

    def K(self, X, X2 = None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(X2, tf.int32), [-1]) if X2 is not None else X
        # graph kernel
        #GL = self.g_var*tf.linalg.expm(- self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        GL = self.g_var*tf.linalg.inv(tf.eye(coord.shape[0], dtype = tf.float64) + self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        return tf.gather(tf.gather(GL, X), X2, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

class transductive(gpflow.kernels.Kernel):
    def __init__(self, lengthscales = 1., variance = 1., g_sigma = 1000., g_var = 1.):
        super().__init__(active_dims=None)
        self.g_sigma = gpflow.Parameter(g_sigma, transform = positive())
        self.g_var = gpflow.Parameter(g_var, transform = positive())
        self.base_kernel = gpflow.kernels.SquaredExponential(lengthscales = lengthscales, variance = variance)
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        #self.sparse_L = tf.cast(convert_sparse_matrix_to_sparse_tensor(nx.laplacian_matrix(G)), tf.float64)
        self.sparse_L = tf.cast(convert_sparse_matrix_to_sparse_tensor(G.L), tf.float64)

    def K(self, X, X2 = None):
        if X2 is None:
            X2 = X
        # graph kernel
        #GL = self.g_var*tf.linalg.expm(self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        GL = (1./self.g_var)*(tf.eye(coord.shape[0], dtype = tf.float64) + self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        inner = tf.eye(coord.shape[0], dtype = tf.float64) + (GL @ self.base_kernel.K(coord, coord))
        K_tranductive = self.base_kernel.K(X, X2) - ( self.base_kernel.K(X, coord) @ tf.linalg.inv(inner) @ GL @ self.base_kernel.K(coord, X2) )
        
        return K_tranductive

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

class GGP(gpflow.kernels.Kernel):
    def __init__(self, lengthscales = 1., variance = 1., base_kernel = 'RBF'):
        super().__init__(active_dims=None)
        if base_kernel == 'RBF':
            self.base_kernel = gpflow.kernels.SquaredExponential(lengthscales = lengthscales, variance = variance)
        elif base_kernel == 'I':
            self.base_kernel = gpflow.kernels.White(variance = variance)
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        self.sparse_A = tf.cast(convert_sparse_matrix_to_sparse_tensor(G.W), tf.float64)
        self.D = tf.linalg.diag(1./(tf.reduce_sum(tf.sparse.to_dense(self.sparse_A), axis = 0) + 1))
        #self.D = tf.linalg.diag([tf.cast(1./(val + 1), tf.float64) for (node, val) in G.degree()])
        self.A = tf.sparse.to_dense(self.sparse_A) + tf.eye(self.sparse_A.shape[0], dtype = tf.float64)
        self.P = self.D @ self.A

    def K(self, X, X2 = None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(X2, tf.int32), [-1]) if X2 is not None else X
        K = self.base_kernel.K(coord)
        PKP = self.P @ K @ tf.transpose(self.P)
        return tf.gather(tf.gather(PKP, X), X2, axis=1)

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

class wavelet(gpflow.kernels.base.Kernel):
    def __init__(self, low_pass=1.0, scales=(1.0/5.0, 2.0/5.0), base_kernel=None, node_feats=None):
        super().__init__()
        self.num_scales = len(scales)
        self.base_kernel = base_kernel
        self.num_nodes = G.N
        if node_feats is None:
            self.node_feats = tf.reshape(tf.range(self.num_nodes, dtype=tf.float64), (-1, 1))
        else:
            self.node_feats = node_feats
        # self.L = tf.cast(G.L.todense(), tf.float64)
        e, U = tf.linalg.eigh(G.L.todense())
        self.e, self.U = tf.cast(e, tf.float64), tf.cast(U, tf.float64)
        self.alpha = gpflow.Parameter(low_pass, transform=positive())
        for attr in range(1, self.num_scales + 1):
            self.__dict__['scale' + str(attr)] = gpflow.Parameter(scales[attr-1],
                                                                  transform=positive())
    def low(self):
        return self.U @ tf.linalg.diag(1. / (1. + self.alpha * self.e)) @ tf.transpose(self.U)
    def band(self, scale=1.):
        # range = np.linspace(0, tf.reduce_max(self.e), 100)
        # range_band = range * scale * np.exp(-scale * range)
        return self.U @ tf.linalg.diag(
            self.e * scale * tf.math.exp(-scale * self.e)) @ tf.transpose(self.U)
    def wavelet(self):
        band = tf.math.reduce_sum([self.band(self.__dict__['scale' + str(attr)]) for attr in
                                   range(1, self.num_scales + 1)], axis=0)
        return band + self.low()
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

def GP(kernel, data, test_data):
    global m
    m = gpflow.models.GPR(
        data, kernel=kernel
    )
    #set_trainable(m.likelihood.variance, False)
    print_summary(m)

    def step_callback(step, variables=None, values=None):
        if step % 10 == 0:
            pred = m.predict_y(test_data)[0]
            print('MAE =', np.mean(np.abs(pred.numpy().flatten() - test_t)))
            print(f"Epoch {step}")
            print_summary(m)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, variables=m.trainable_variables, step_callback = step_callback)
    print_summary(m)

    pred = m.predict_y(test_data)[0]
    mae = np.mean(np.abs(pred.numpy().flatten() - test_t))
    plot(pred, mae)
    print('MAE =', mae)

if __name__ == '__main__':
    #kernel = gpflow.kernels.SquaredExponential(lengthscales = 10., variance = 1.)
    if model in ['gp', 'transductive']:
        data_pair = (data, t.reshape(-1,1))

    elif model in ['graph_only', 'ggp', 'wavelet']:
        train_id, test_data = np.arange(training, dtype=np.float).reshape(-1,1), np.arange(training, G.N, dtype=np.float).reshape(-1,1)
        data_pair = (train_id, t.reshape(-1,1))

    if model == 'gp':
        kernel = gpflow.kernels.SquaredExponential(lengthscales = 1., variance = 1.)
    elif model == 'graph_only':
        kernel = graph_diff(g_sigma = 100., g_var = 1.)
    elif model == 'transductive':
        kernel = transductive(lengthscales = 10., variance = 1., g_sigma = 1000., g_var = 1.)
    elif model == 'ggp':
        kernel = GGP(lengthscales = 1., variance = 1.)
    elif model == 'wavelet':
        kernel = wavelet(low_pass=1., scales=[1., 100.], base_kernel = gpflow.kernels.SquaredExponential(lengthscales = 1., variance = 1.), node_feats = coord)
    
    GP(kernel, data_pair, test_data)
