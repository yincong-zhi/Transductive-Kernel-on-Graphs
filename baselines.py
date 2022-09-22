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

from kernels.chebyshev_kernel import Chebyshev
from kernels.ggp_kernel import GraphGP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default="ggp", type=str, help='ggp. chebyshev')
parser.add_argument("--opt", default='adam', type=str, help="scipy, adam")
parser.add_argument("--data", default='Cora', type=str, help="Cora, Citeseer, Texas, Wisconsin, Cornell, Chameleon, Squirrel")
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

if parser.kernel == "chebyshev":
    #kernel = Chebyshev(L_normalized, poly_degree=poly_degree, base_kernel=base_kernel,node_feats=node_feats)
    pass
elif parser.kernel == "ggp":
    kernel = GraphGP(adj, base_kernel=gpflow.kernels.Polynomial(), node_feats=allx)

data = (tf.cast(tf.reshape(train_idx, (-1, 1)), tf.float64), tf.cast(y, dtype=tf.float64))
num_classes = y.max().numpy() + 1
invlink = gpflow.likelihoods.RobustMax(num_classes)  # Robustmax inverse link function
likelihood = gpflow.likelihoods.MultiClass(num_classes, invlink=invlink)  # Multiclass likelihood

m = gpflow.models.VGP(data, likelihood=likelihood, kernel=kernel, num_latent_gps=num_classes)
opt = tf.optimizers.Adam(lr=0.1)

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

def optimize_tf(model, step_callback, lr=0.1):
    opt = tf.optimizers.Adam(lr=lr)
    for epoch_idx in range(300):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            loss = model.training_loss()
            gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        step_callback(epoch_idx)
        #print(f"{epoch_idx}:\tLoss={loss}")
        
optimize_tf(m, step_callback, lr = 0.1)