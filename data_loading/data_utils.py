import numpy as np


def subsample_neighbors(node_idcs, sample_sizes, adj_list_mat):
    """
    Given a set of node indices, this method samples a fixed number of
    neighbors for each of the nodes. Depending on the input, it may
    recursively sample a fixed number of neighbors for the nodes that have
    just been sampled.
    :param node_idcs: Numpy array of indices of central nodes for which
    neighbors should be sampled.
    :param sample_sizes: Tuple or list of integers. The first entry
    specifies the number of neighbors to sample from the one-hop
    neighborhood, the second entry specifies the number of neighbors to
    sample from the two-hop neighborhood etc.
    :param adj_list_mat: Array of shape [N, max_degree] listing for each
    node the indices of its neighboring nodes.
    :return:
    """
    sample_size = sample_sizes[0]
    neighbors = adj_list_mat[node_idcs]
    # shuffle columns
    np.random.shuffle(neighbors.T)
    sampled_neighbors = neighbors[:, :sample_size]  # Shape [B, sample_size]
    subgraph_node_idcs = np.unique(np.concatenate(
            (node_idcs, sampled_neighbors.reshape(-1)), axis=0))
    if len(sample_sizes) > 1:
        return subsample_neighbors(subgraph_node_idcs, sample_sizes[1:],
                                   adj_list_mat)
    return subgraph_node_idcs


def get_adj_list_matrix(adj_matrix):
    """
    Given a NxN SciPy sparse adjacency matrix returns a dense adjacency list
    matrix of shape [N, max_degree] where each row i contains the node indices
    of neighbors of i. If node i has less than max_degree neighbors, some of
    its neighbors are randomly resampled.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 0.0
    max_degree = int(np.max(np.sum(adj_matrix, axis=0)))
    adj_matrix.eliminate_zeros()
    adj_matrix = adj_matrix.tolil()
    rows = []
    for idx, neighbors in enumerate(adj_matrix.rows):
        if len(neighbors) == 0:     # isolated node
            rows.append(np.array([idx for _ in range(max_degree)]))
            continue
        num_additional = int(max_degree - len(neighbors))
        add_samples = np.random.choice(neighbors, size=num_additional)
        row = np.concatenate((neighbors, add_samples), axis=0)
        np.random.shuffle(row)
        rows.append(row)
    adj_list_mat = np.array(rows)
    return adj_list_mat


def get_subgraph_matrix(adj_matrix, num_hops):
    """
    Given a NxN SciPy sparse adjacency matrix, computes a NxN dense matrix
    that indicates for each node its neighbors in the K hop neighborhood,
    where K=num_hops.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    power = adj_matrix
    for hop in range(num_hops - 1):
        power = power.dot(adj_matrix)
    subgraph_matrix = np.array(power.todense())
    subgraph_matrix[subgraph_matrix > 0.0] = 1.0
    subgraph_matrix = subgraph_matrix.astype(np.bool)
    return subgraph_matrix


def get_subgraph_matrix_sparse(adj_matrix, num_hops):
    """
    Same as `get_subgraph_matrix` but only returns the indices of shape [N, 2] of non-zero indices
    in the subgraph matrix.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    power = adj_matrix
    for hop in range(num_hops - 1):
        power = power.dot(adj_matrix)
    power = power.tocoo()
    return np.stack((power.row, power.col), axis=-1)