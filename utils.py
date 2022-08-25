"""
Data loading methods are adapted from
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
import math
import pathlib
import pickle as pk

import pygsp
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 15}
matplotlib.rc('font', **font)


def sparse_mat_to_sparse_tensor(sparse_mat):
    """
    Converts a scipy csr_matrix to a tensorflow SparseTensor.
    """
    coo = sparse_mat.tocoo()
    indices = np.stack([coo.row, coo.col], axis=-1)
    tensor = tf.sparse.SparseTensor(indices, sparse_mat.data, sparse_mat.shape)
    return tensor


def dense_tensor_to_sparse_tensor(dense_tensor):
    """
    Converts a scipy csr_matrix to a tensorflow SparseTensor.
    """
    indices = tf.where(tf.not_equal(dense_tensor, 0))
    sparse_tensor = tf.SparseTensor(indices, tf.gather_nd(dense_tensor, indices), dense_tensor.shape)
    return sparse_tensor


def dense_matrix_to_sparse_matrix(dense_matrix):
    """
    Converts a two-dimensional NumPy array to a SciPy csr_matrix.
    """
    indices = np.where(dense_matrix > 0.0)
    sparse_matrix = sp.csr_matrix((np.ones(len(indices[0])),
                                   (indices[0], indices[1])),
                                  shape=(dense_matrix.shape[0], dense_matrix.shape[1]))
    return sparse_matrix


def sparse_matrix_to_parts(sparse_mat):
    """
    Extracts indices-array and values-array from a scipy sparse csr_matrix.
    :return: Two dense NumPy arrays:
    - indices: Indices of the non-zero entries in the sparse matrix. Shape
    [E, 2].
    - vals: Values corresponding to the edges. Shape [E].
    """
    temp = sparse_mat.tocoo()
    indices = np.stack((temp.row, temp.col), axis=-1)
    vals = temp.data
    return indices, vals


def normalize_adj(adj):
    """Symmetrically normalize sparse adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_dense(adj):
    """Symmetrically normalize dense adjacency matrix."""
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.matmul(d_mat_inv_sqrt, np.matmul(adj, d_mat_inv_sqrt))


def get_submatrix(adj_matrix, node_idcs, num_hops=2):
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    power = adj_matrix
    for hop in range(num_hops-1):
        power = power.dot(adj_matrix)
    sub_mat = power[node_idcs, :].tocoo()
    rel_node_idcs = np.unique(sub_mat.col)
    return rel_node_idcs


def get_conv_matrix(adj_matrix, float_type=None, scipy_sparse=False):
    """Computes the convolution matrix from the adjacency matrix. The
    convolution matrix is just the degree normalised adjacency matrix with
    self-loops.
    WARNING: This method modifies the adjacency matrix.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    conv_matrix = normalize_adj(adj_matrix)
    if scipy_sparse is True:
        return conv_matrix
    temp = conv_matrix.tocoo()
    conv_mat_idcs = np.stack((temp.row, temp.col), axis=-1)
    conv_mat_vals = temp.data
    if float_type is not None:
        conv_mat_vals = conv_mat_vals.astype(float_type)
    return conv_mat_idcs, conv_mat_vals


def permutation_without_cycles(length):
    permutation = np.random.choice(length, length, replace=False)

    cycle_ids = (permutation == np.arange(length))
    num_cycles = np.sum(cycle_ids)
    while num_cycles > 1:
        sub_perm = permutation_without_cycles(num_cycles)
        permutation[cycle_ids] = permutation[cycle_ids][sub_perm]
        cycle_ids = (permutation == np.arange(length))
        num_cycles = np.sum(cycle_ids)
    # swap with element not at its original position
    if num_cycles == 1:
        cycle_idx = np.where(cycle_ids)[0][0]
        for swap_idx in range(length):
            if permutation[swap_idx] != cycle_idx:
                temp = permutation[swap_idx]
                permutation[swap_idx] = permutation[cycle_idx]
                permutation[cycle_idx] = temp
                break
    # sanity check
    cycle_ids = (permutation == np.arange(length))
    assert np.sum(cycle_ids) == 0
    return permutation


def sample_random_edges(num_nodes, num_edges, rstate=None):
    """
    Samples random edges for a given number of nodes such that the resulting
    graph is connected.
    :return: Edge indices of shape [num_edges, 2]
    """
    if num_nodes > num_edges:
        raise ValueError(f"Number of nodes ({num_nodes}) is higher than number"
                         f"of edges to be sampled ({num_edges}). This will "
                         "form a disconnected graph.")
    perm_f = np.random.permutation if rstate is None else rstate.permutation
    current_edges = set()
    while len(current_edges) < num_edges:
        remaining_req_edges = num_edges - len(current_edges)
        origin_nodes = np.arange(num_nodes)
        destination_nodes = permutation_without_cycles(num_nodes)

        p = perm_f(num_nodes)
        origin_nodes, destination_nodes = (origin_nodes[p], destination_nodes[p])
        origin_nodes, destination_nodes = (origin_nodes[:remaining_req_edges],
                                           destination_nodes[:remaining_req_edges])
        current_edges |= set({(n1, n2) if n1 < n2 else (n2, n1) for n1, n2
                             in zip(origin_nodes, destination_nodes)})
    assert len(current_edges) == num_edges
    left, right = zip(*current_edges)
    edge_indices = np.stack((np.array(left), np.array(right)), axis=1)
    return edge_indices


def compute_batch_size(num_train_edges):
    # we want at least ten batches
    batch_size = num_train_edges // 10
    # round to next largest power of 2
    power = math.ceil(math.log2(batch_size))
    batch_size = int(2**power)
    return min(batch_size, 1024)


def save_tf_module_weights(module, filepath):
    filepath = pathlib.Path(filepath)
    param_vals = [param.numpy() for param in module.trainable_parameters]
    with filepath.open("wb") as fd:
        pk.dump(param_vals, fd)


def load_tf_module_weights(module, filepath):
    filepath = pathlib.Path(filepath)
    with filepath.open("rb") as fd:
        param_vals = pk.load(fd)
    for param, val in zip(module.trainable_parameters, param_vals):
        if isinstance(param.transform, tfp.bijectors.Sigmoid):  # addresses numerical issue with Sigmoid transform
            val[val == 1.0] = 1.0 - 1e-5
        param.assign(val)


def compute_kl_scale(logger, cold_posterior_period):
    if "stop_cold" in logger.misc_info:
        kl_scale = ((logger.current_epoch - logger.misc_info["stop_cold"])
                    / cold_posterior_period
                    if cold_posterior_period > 0 else 1.0)
        kl_scale = max(0.0, min(kl_scale, 1.0))
    else:
        if logger.reached_relative_reduction("train_likelihood", 0.1):
            print("End of cold phase")
            logger.misc_info["stop_cold"] = logger.current_epoch
        kl_scale = 0.0
    return kl_scale


def apply_batched_scaler(data, scaler_f):
    original_shape = data.shape
    data = data.reshape(-1, original_shape[-1])
    data = scaler_f(data)
    data = data.reshape(*original_shape)
    return data


def get_subset_cluster_entropy(subset_idcs, nx_graph, cluster_level=0):
    """
    Computes the entropy of a subset of nodes with respect to their belonging to a certain
    cluster in the graph.
    :param subset_idcs: NumPy array with node indices.
    :param nx_graph: NetworkX graph with node labels of format "3-2-4", where first integer
    denotes top-level cluster, second integer denotes second-level cluster etc.
    """
    cluster_labels = np.array(["-".join(label.split("-")[:cluster_level+1]) for label in nx_graph.nodes])
    unique_cluster_labels = np.sort(np.unique(cluster_labels))
    cluster_idcs = np.array([np.where(unique_cluster_labels == label)[0][0] for label in cluster_labels])
    train_cluster_idcs = cluster_idcs[subset_idcs.numpy()]
    rel_freqs = st.relfreq(train_cluster_idcs, numbins=np.max(cluster_idcs)+1).frequency
    entropy = st.entropy(rel_freqs)
    return entropy


def plot_cumulative_spectral_density(adj_matrix, graph_name, plot_eigvals=False,
                                     plot_spectrum_rank=True, normalized=True):
    if np.abs(adj_matrix - adj_matrix.T).sum() > 1e-3:    # PyGSP does not support computing normalized Laplacian for directed graphs
        L = laplacian(adj_matrix, normed=True).toarray()
    else:
        graph = pygsp.graphs.graph.Graph(adj_matrix)
        if normalized:
            graph.compute_laplacian("normalized")
        else:
            graph.compute_laplacian()
        L = graph.L.toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.real(eigvals)
    ls = np.linspace(0.0, np.max(eigvals), num=2*L.shape[0])
    cumul_density = []
    for l in ls:
        cumul_density.append(np.sum(eigvals < l) / L.shape[0])
    cumul_density = np.array(cumul_density)
    plt.title(f"Spectral density of {graph_name}")
    plt.xlabel("eigenvalues")
    plt.ylabel("cumulative spectral density")
    plt.plot(ls, cumul_density)
    if plot_eigvals:
        plt.plot(eigvals, np.zeros_like(eigvals), 'x')
    plt.show()

    if plot_spectrum_rank:
        plt.figure(figsize=(8, 4.5))
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.scatter(eigvals, np.arange(len(eigvals)))
        plt.title(f"Ranked spectrum of {graph_name}")
        plt.xlabel("eigenvalues")
        plt.ylabel("rank index")
        plt.savefig(f"ranked_spectrum_{graph_name.lower()}.pdf")
        plt.show()
    return eigvals, eigvecs


def layout_plots():
    font = {'size': 15}
    matplotlib.rc('font', **font)


if __name__ == '__main__':
    # print(sample_random_edges(10, 20))
    pass