import numpy as np
import scipy as sp
import networkx as nx
import tensorflow as tf
#import warnings
#warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow
import matplotlib.pyplot as plt
#matplotlib inline
import gpflow
from gpflow.utilities import print_summary, set_trainable, positive
from gpflow.ci_utils import ci_niter
from pygsp import graphs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default="graph_poly", type=str, help='polynomial, matern12, matern32, matern52, transdutive, wavelet, graph_poly')
parser.add_argument("--opt", default='scipy', type=str, help="scipy, adam")
parser.add_argument("--data", default='Cora', type=str, help="Cora, Citeseer, Texas, Wisconsin, Cornell, Chameleon, Squirrel")
parser.add_argument("--poly", default=[0., 1., 0., 0., 0.], type=float, nargs="+", help="polynomial coefficients")
parser.add_argument("--scipy_max", default=50, type=float, help="maximum # of iterations for scipy optimizer")
parser.add_argument('--train_on_val', type=bool, default=False, help='If True, validation set is included in the training')
parser = parser.parse_args()

import torch
import torch_geometric.datasets as datasets
if parser.data in ["Cora", "Citeseer", "PubMed"]:
    data_name = parser.data
    dataset = datasets.Planetoid(root=f'/tmp/{data_name}', name=data_name)
    data = dataset.data

    if parser.train_on_val:
        data.train_mask += data.val_mask
    x, y, tx, ty, vx, vy, allx, ally = data.x[data.train_mask], data.y[data.train_mask], data.x[data.test_mask], data.y[data.test_mask], data.x[data.val_mask], data.y[data.val_mask], data.x, data.y
    x_train_test = tf.cast(allx, tf.float64)
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    train_idx, val_idx, test_idx = tf.where(train_mask), tf.where(val_mask), tf.where(test_mask)
    train_idx, val_idx, test_idx = tf.reshape(train_idx, -1), tf.reshape(val_idx, -1), tf.reshape(test_idx, -1)
    adj = torch.zeros((data.num_nodes, data.num_nodes))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    G = graphs.Graph(adj.numpy())
    G.compute_laplacian('normalized')
else:
    from data_loading.node_prediction_dataset import NodePredictionDataset
    ds = NodePredictionDataset(parser.data, "")

    adj, features, labels, train_idx, val_idx, test_idx, edge_index, edge_weights = ds.get_full_training_data()
    if parser.train_on_val:
        train_idx = np.concatenate((train_idx, val_idx))
    if not (adj.todense() == adj.todense().T).all():
        adj = 0.5 * (adj + adj.T)
    x, y, tx, ty, vx, vy, allx, ally = features[train_idx], labels[train_idx], features[test_idx], labels[test_idx], features[val_idx], labels[val_idx], features, labels
    x_train_test = tf.cast(allx, tf.float64)
    G = graphs.Graph(adj.todense())
    G.compute_laplacian('normalized')

# one-hot encode for regression
y_hot = np.zeros((y.shape[0], y.max()+1))
for i in range(y.max()+1):
    y_hot[:,i] = y == i

y_val_hot = np.zeros((vy.shape[0], y.max()+1))
for i in range(y.max()+1):
    y_val_hot[:,i] = vy == i

y_test_hot = np.zeros((ty.shape[0], y.max()+1))
for i in range(y.max()+1):
    y_test_hot[:,i] = ty == i

'''Transductive GP'''

class Transductive(gpflow.kernels.Kernel):
    def __init__(self, offset = 1., variance = 1., g_sigma = 1., g_var = 1.):
        super().__init__(active_dims=None)
        self.g_sigma = gpflow.Parameter(g_sigma, transform = positive())
        self.g_var = gpflow.Parameter(g_var, transform = positive())
        self.base_kernel = gpflow.kernels.Matern32(lengthscales = offset, variance = variance)
        #self.base_kernel = gpflow.kernels.Polynomial(offset = offset, variance = variance)
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
        #GL = (1./self.g_var)*tf.linalg.expm(self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        GL = (1./self.g_var)*(tf.eye(x_train_test.shape[0], dtype = tf.float64) + self.g_sigma * tf.sparse.to_dense(self.sparse_L))
        inner = tf.eye(x_train_test.shape[0], dtype = tf.float64) + (GL @ self.base_kernel.K(x_train_test, x_train_test))
        K_tranductive = self.base_kernel.K(X, X2) - ( self.base_kernel.K(X, x_train_test) @ tf.linalg.inv(inner) @ GL @ self.base_kernel.K(x_train_test, X2) )
        
        return K_tranductive

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

'''Wavelet Transductive GP'''
class Wavelet(gpflow.kernels.Kernel):
    def __init__(self, offset = 1., variance = 1., alpha = 1., scales = [1., 2.], g_var = 1.):
        super().__init__(active_dims=None)
        self.g_var = gpflow.Parameter(g_var, transform = positive())
        self.alpha = gpflow.Parameter(alpha, transform = positive())
        for attr in range(1, len(scales) + 1):
            self.__dict__['scale' + str(attr)] = gpflow.Parameter(scales[attr-1], transform=positive())
        self.num_scales = len(scales)
        #self.base_kernel = gpflow.kernels.Polynomial(offset = offset, variance = variance)
        self.base_kernel = gpflow.kernels.Matern12(lengthscales = offset, variance = variance)
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        e, U = tf.linalg.eigh(G.L.todense())
        #e, U = tf.linalg.eigh(nx.laplacian_matrix(G).todense())
        self.e, self.U = tf.cast(e, tf.float64), tf.cast(U, tf.float64)

    def low(self, x, alpha=1.):
        return 1. / (1. + alpha * x)

    def mexican_hat_normalized(self, x, scale=1.):
        x = x * scale
        pi = tf.convert_to_tensor(np.pi, dtype=x.dtype)
        const = tf.cast(2.0 * tf.sqrt(2.0 / 3.0), dtype=x.dtype)
        const *= tf.cast(tf.pow(pi, -1.0 / 4.0), dtype=x.dtype)
        y = const * x**2 * tf.exp(-0.5 * x ** 2)
        return y
    
    def wavelet(self, x):
        wave = self.low(x, self.alpha)
        for i in range(self.num_scales):
            wave += self.mexican_hat_normalized(x, self.__dict__['scale' + str(i+1)])
        return wave

    def K(self, X, X2 = None):
        if X2 is None:
            X2 = X
        # wavelet kernel
        GL = (1./self.g_var) * self.U @ tf.linalg.diag(1./(self.wavelet(self.e))) @ tf.transpose(self.U)
        inner = tf.eye(x_train_test.shape[0], dtype = tf.float64) + GL @ self.base_kernel.K(x_train_test, x_train_test)
        K_tranductive = self.base_kernel.K(X, X2) - ( self.base_kernel.K(X, x_train_test) @ tf.linalg.inv(inner) @ GL @ self.base_kernel.K(x_train_test, X2) )
        return K_tranductive

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

'''Polynomial Transductive GP'''
class Polynomial(gpflow.kernels.Kernel):
    def __init__(self, base_param = [1., 1.], g_var = 1., beta = [1., -1., 0.]):
        super().__init__(active_dims=None)
        self.g_var = gpflow.Parameter(g_var, transform = positive())
        self.degree = len(beta)
        for attr in range(len(beta)):
            self.__dict__['beta' + str(attr)] = gpflow.Parameter(beta[attr])
        self.base_kernel = gpflow.kernels.Matern12(lengthscales = base_param[0], variance = base_param[1])
        #self.base_kernel = gpflow.kernels.Polynomial(offset = base_param[0], variance = base_param[1])
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        graph = graphs.Graph(adj)
        graph.compute_laplacian('normalized')
        e, U = tf.linalg.eigh(graph.L.todense())
        #e, U = tf.linalg.eigh(nx.laplacian_matrix(graph).todense())
        self.e, self.U = tf.cast(e, tf.float64), tf.cast(U, tf.float64)

    def poly(self):
        poly = tf.math.reduce_sum([self.__dict__['beta' + str(attr)] * (self.e ** attr) for attr in range(self.degree)], axis = 0)
        return poly

    def sigmoid(self):
        #sigmoid = 1./(1. + tf.math.exp(-self.poly()))
        sigmoid = tf.math.softplus(self.poly())
        return 1. + sigmoid

    def K(self, X, X2 = None):
        if X2 is None:
            X2 = X
        GL = (1./self.g_var) * self.U @ tf.linalg.diag(self.sigmoid()) @ tf.transpose(self.U)
        inner = tf.eye(x_train_test.shape[0], dtype = tf.float64) + GL @ self.base_kernel.K(x_train_test, x_train_test)
        K_tranductive = self.base_kernel.K(X, X2) - ( self.base_kernel.K(X, x_train_test) @ tf.linalg.inv(inner) @ GL @ self.base_kernel.K(x_train_test, X2) )
        return K_tranductive

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

training_loss = []
valid_accs = []
test_accs = []

def step_callback(step, variables=None, values=None):
    # test acc
    y_pred = m.predict_y(tf.cast(tx, tf.float64))[0]
    y_class = tf.argmax(y_pred, 1)
    test_acc = (tf.reduce_sum(tf.cast(y_class == ty, tf.float64))/test_idx.shape[0]).numpy()
    # validation acc
    y_pred_val = m.predict_y(tf.cast(vx, tf.float64))[0]
    y_class_val = tf.argmax(y_pred_val, 1)
    val_acc = (tf.reduce_sum(tf.cast(y_class_val == vy, tf.float64))/val_idx.shape[0]).numpy()
    print(f'epoch = {step}', 'val acc =', val_acc, 'test acc =', test_acc)
    training_loss.append(m.training_loss().numpy())
    valid_accs.append(val_acc)
    test_accs.append(test_acc)
    if step % 10 == 0:
        print_summary(m)
        #plt.plot(kernel.e, kernel.sigmoid())
        #plt.plot(kernel.e, 1./kernel.sigmoid())
        #plt.legend([r'$r(\lambda)$', r'$k(\lambda)$'])
        #plt.show()

if __name__ == '__main__':
    if parser.kernel == 'polynomial':
        kernel = gpflow.kernels.Polynomial()
    elif parser.kernel == 'matern52':
        kernel = gpflow.kernels.Matern52()
    elif parser.kernel == 'matern32':
        kernel = gpflow.kernels.Matern32()
    elif parser.kernel == 'matern12':
        kernel = gpflow.kernels.Matern12()
    elif parser.kernel == 'transductive':
        kernel = Transductive(offset = 1., variance = 1., g_sigma = 300., g_var = 1.)
    elif parser.kernel == 'wavelet':
        kernel = Wavelet(offset = 1., variance = 1., alpha = 300., scales = [0.8], g_var = 1.)
    elif parser.kernel == 'graph_poly':
        kernel = Polynomial(base_param = [1., 1.], g_var = 0.05, beta = parser.poly)

    m = gpflow.models.GPR(
        (tf.cast(x, dtype = tf.float64), y_hot), kernel=kernel
    )

    if parser.opt == 'scipy':
        opt = gpflow.optimizers.Scipy()
        opt.minimize(m.training_loss, variables=m.trainable_variables, step_callback = step_callback, options=dict(maxiter=ci_niter(parser.scipy_max)))
        pred = tf.math.argmax(m.predict_f(tf.cast(tx, dtype = tf.float64))[0], axis = 1)
        print('% accuracy =', tf.reduce_sum(tf.cast(pred == ty, tf.float64)).numpy()/ty.shape[0] * 100.)

    elif parser.opt == 'adam':
        def optimize_tf(model, step_callback, lr=0.1):
            opt = tf.optimizers.Adam(lr=lr)
            for epoch_idx in range(300):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(model.trainable_variables)
                    loss = model.training_loss()
                    gradients = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(gradients, model.trainable_variables))
                step_callback(epoch_idx)
        optimize_tf(m, step_callback, lr=0.1)
        pred = tf.math.argmax(m.predict_f(tf.cast(tx, dtype = tf.float64))[0], axis = 1)
        print('% accuracy =', tf.reduce_sum(tf.cast(pred == ty, tf.float64)).numpy()/ty.shape[0] * 100.)

