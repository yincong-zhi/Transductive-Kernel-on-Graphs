import pathlib
import pickle as pk
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets as datasets
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from data_loading.data_utils import get_adj_list_matrix, get_subgraph_matrix_sparse
from utils import normalize_adj, get_conv_matrix, plot_cumulative_spectral_density


class NodePredictionDataset:
    def __init__(self, dataset_name, dataset_folder, split_seed=None, split_idx=0,
                 tfidf_transform=True, standardize=False, abs_idcs=False,
                 training_samples_per_class=None, missing_fraction=None, train_frac=None,
                 corrupt_seed=None, float_type=np.float32):
        """
        :param abs_idcs: If True, the data loaders also return the absolute
        indices of the central nodes in the batch.
        :param training_samples_per_class: Specifies the number of training
        samples per class. Random training samples per class are discarded
        until the desired number is reached.
        :param split_idx: Some datasets come with multiple pre-defined train-val-test splits. This
        parameter is used to choose the split.
        :param missing_fraction: Probability of each feature being replaced by
        a zero vector.
        """
        rstate = np.random.RandomState(split_seed)
        dataset_folder = pathlib.Path(dataset_folder)
        self.abs_idcs = abs_idcs

        # Load raw data set
        dataset_path = dataset_folder / f"ptgeo_{dataset_name}/"
        if dataset_name in ["Cora", "Citeseer", "PubMed"]:
            dataset = datasets.Planetoid(dataset_path, dataset_name)
        elif dataset_name in ["Computers", "Photo"]:
            dataset = datasets.Amazon(dataset_path, dataset_name)
        elif dataset_name in ["Physics", "CS"]:
            dataset = datasets.Coauthor(dataset_path, dataset_name)
        elif dataset_name in ["Texas", "Cornell", "Wisconsin"]:
            dataset = datasets.WebKB(dataset_path, dataset_name)
        elif dataset_name in ["Chameleon", "Squirrel"]:
            dataset = datasets.WikipediaNetwork(dataset_path, dataset_name)
        elif dataset_name in ["Actor"]:
            dataset = datasets.Actor(dataset_path, dataset_name)
        else:
            raise ValueError(f"No data set found for name {dataset_name}.")
        data = dataset.data
        features = data.x.numpy()
        labels = data.y.numpy()

        # Data set properties
        num_nodes = len(labels)
        num_classes = len(np.unique(labels))
        num_train = (np.sum(data.train_mask.numpy())
                     if "train_mask" in data.keys else num_classes * 20)
        num_val = (np.sum(data.val_mask.numpy())
                   if "val_mask" in data.keys else num_classes * 30)
        num_test = (np.sum(data.test_mask.numpy())
                    if "test_mask" in data.keys
                    else num_nodes - num_train - num_val)
        assert num_train % num_classes == 0, "Ill-specified number of samples per class"
        num_train_per_class = num_train // num_classes

        # Train, val, test split
        if "train_mask" not in data.keys or split_seed is not None:
            print("INFO: Generating custom random train/val/test split")
            num_val_per_class = num_val // num_classes
            train_idx = _get_random_subset_per_class(num_train_per_class, labels, set(), rstate)
            val_idx = _get_random_subset_per_class(num_val_per_class, labels, train_idx, rstate)
            test_idx = set(range(num_nodes)) - train_idx - val_idx
            assert (train_idx.isdisjoint(val_idx)
                    and train_idx.isdisjoint(test_idx)
                    and val_idx.isdisjoint(test_idx))
            train_idx = np.array(sorted(list(train_idx)))
            val_idx = np.array(sorted(list(val_idx)))
            test_idx = np.array(sorted(list(test_idx)))
        else:
            print("INFO: Using pre-defined train/val/test split")
            def get_idcs(mask, split_idx):
                mask = mask[:, split_idx] if len(mask.shape) > 1 else mask
                return np.where(mask)[0]
            train_idx = get_idcs(data.train_mask.numpy(), split_idx)
            val_idx = get_idcs(data.val_mask.numpy(), split_idx)
            test_idx = get_idcs(data.test_mask.numpy(), split_idx)

        if train_frac is not None:
            training_samples_per_class = int(round(train_frac * np.sum(labels[train_idx] == 0)))
        # If necessary, throw away some training samples
        if training_samples_per_class is not None:
            print("INFO: Throwing away some training samples")
            train_idx = _cap_samples_per_class(train_idx, labels, training_samples_per_class,
                                               rstate)

        # Create adjacency matrix
        adj_idcs = data.edge_index.numpy().T    # Shape [num_nodes, 2]
        edge_index = data.edge_index
        edge_weights = data.edge_attr
        if corrupt_seed is not None:
            corrupt_rstate = np.random.RandomState(corrupt_seed)
            adj_idcs, edge_index, edge_weights = self.get_corrupted_adj_information(
                corrupt_rstate, adj_idcs, edge_index, edge_weights)

        # Transform features if necessary
        if dataset_name.lower() != 'pubmed' and tfidf_transform:  # tf-idf transform features, unless it's pubmed, which already comes with tf-idf
            print("INFO: Applying TDIDF transformation")
            transformer = TfidfTransformer(smooth_idf=True)
            features = transformer.fit_transform(features).todense()
            features = np.array(features)

        self.std_scaler = None
        if standardize is True:
            self.std_scaler = StandardScaler()
            features = self.std_scaler.fit_transform(features)

        # If specified, set some node features to 0.
        if missing_fraction is not None:
            zero_idcs = rstate.choice(num_nodes, int(num_nodes*missing_fraction))
            features[zero_idcs] = 0.0

        # Assign to attributes and convert if necessary
        self.num_nodes = features.shape[0]
        self.num_classes = np.max(labels)+1
        self.features = features.astype(float_type)
        self.labels = labels
        self.adj_idcs = adj_idcs                        # Shape [num_nodes, 2]
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self._float_type = float_type
        self._tf_float_type = (tf.float64 if float_type == np.float64 else tf.float32)

        # Variables filled on a per-need basis
        self.subgraph_matrix_idcs = None
        self.adj_list_matrix = None
        self.subsample_sizes = None
        self.conv_mat_idcs = None
        self.conv_mat_vals = None

    def get_full_training_data(self):
        adj_matrix = self.get_adj_matrix()
        return adj_matrix, self.features, self.labels, self.train_idx, self.val_idx, self.test_idx, self.edge_index, self.edge_weights

    def persist_full_training_data(self, file_path):
        adj_mat, node_feats, labels, train_idx, val_idx, test_idx = self.get_full_training_data()
        file_path = pathlib.Path(file_path)
        with file_path.open("wb") as fd:
            pk.dump({
                "adj_mat": adj_mat,
                "node_feats": node_feats,
                "labels": labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }, fd)

    def _batch_datasets(self, train_dataset, val_dataset, test_dataset, train_batch_size=None,
                        val_batch_size=None, test_batch_size=None):
        """
        Takes train, validation, and test datasets as well as corresponding batch sizes and returns
        batched versions of the datasets. The training dataset is also shuffled.
        """
        train_dataset = train_dataset.shuffle(len(self.train_idx))
        train_batch_size = (len(self.train_idx) if train_batch_size is None else train_batch_size)
        val_batch_size = (len(self.val_idx) if val_batch_size is None else val_batch_size)
        test_batch_size = (len(self.test_idx) if test_batch_size is None else test_batch_size)
        train_dataset = train_dataset.batch(train_batch_size)
        val_dataset = val_dataset.batch(val_batch_size)
        test_dataset = test_dataset.batch(test_batch_size)
        return train_dataset, val_dataset, test_dataset

    def get_data_loaders_unstructured(self, train_batch_size=None, val_batch_size=None,
                                      test_batch_size=None):
        """
        Returns train, validation, and test data loaders without any graph information. Each data
        loader returns a batch of node indices, node features, and labels.
        """
        labels = self.labels.reshape(-1, 1)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_idx, self.features[self.train_idx], labels[self.train_idx]))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.val_idx, self.features[self.val_idx], labels[self.val_idx]))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.test_idx, self.features[self.test_idx], labels[self.test_idx]))
        return self._batch_datasets(train_dataset, val_dataset, test_dataset, train_batch_size,
                                    val_batch_size, test_batch_size)

    def get_data_loaders(self, num_hops, subsample_sizes=None, train_batch_size=None,
                         val_batch_size=None, test_batch_size=None, train_on_val=False):
        subsample = (subsample_sizes is not None)
        adj_matrix = self.get_adj_matrix()
        if subsample is True:
            self.adj_list_matrix = get_adj_list_matrix(adj_matrix)
            self.subsample_sizes = subsample_sizes

        self.subgraph_matrix_idcs = get_subgraph_matrix_sparse(adj_matrix, num_hops)
        # Compute conv matrix
        self.conv_mat_idcs, self.conv_mat_vals = get_conv_matrix(adj_matrix, self._float_type)

        # If specified, merge training and validation set
        train_idx = np.concatenate([self.train_idx, self.val_idx], axis=0) if train_on_val else self.train_idx

        train_dataset = tf.data.Dataset.from_tensor_slices(train_idx)
        val_dataset = tf.data.Dataset.from_tensor_slices(self.val_idx)
        test_dataset = tf.data.Dataset.from_tensor_slices(self.test_idx)
        (train_dataset, val_dataset, test_dataset) = self._batch_datasets(
            train_dataset, val_dataset, test_dataset, train_batch_size,
            val_batch_size, test_batch_size)

        mapping_fn = (self._tf_get_subsampled_subgraph_information if subsample
                      else self._tf_get_subgraph_information)
        train_dataset = train_dataset.map(mapping_fn)
        val_dataset = val_dataset.map(mapping_fn)
        test_dataset = test_dataset.map(mapping_fn)

        return train_dataset, val_dataset, test_dataset

    def _tf_get_subgraph_information(self, idcs):
        [X_batch, y_batch, conv_matrix_idcs, conv_matrix_vals, center_idcs,
         num_subgraph_nodes] = tf.py_function(
            NodePredictionDataset.get_subgraph_information,
            [idcs, self.subgraph_matrix_idcs, self.features, self.labels,
             self.conv_mat_idcs, self.conv_mat_vals],
            [self._tf_float_type, tf.int64, tf.int64, self._tf_float_type, tf.int64, tf.int64]
        )
        conv_matrix = tf.SparseTensor(conv_matrix_idcs, conv_matrix_vals,
                                      (num_subgraph_nodes, num_subgraph_nodes))
        if self.abs_idcs:
            return idcs, X_batch, y_batch, conv_matrix, center_idcs
        return X_batch, y_batch, conv_matrix, center_idcs

    def _tf_get_subsampled_subgraph_information(self, idcs):
        [X_batch, y_batch, conv_matrix_idcs, conv_matrix_vals, center_idcs,
         num_subgraph_nodes] = tf.py_function(
            NodePredictionDataset.get_subsampled_subgraph_information,
            [idcs, self.adj_list_matrix, self.features, self.labels,
             self.adj_idcs, self.subsample_sizes],
            [self._tf_float_type, tf.int64, tf.int64, self._tf_float_type, tf.int64, tf.int64]
        )
        conv_matrix = tf.SparseTensor(conv_matrix_idcs, conv_matrix_vals,
                                      (num_subgraph_nodes, num_subgraph_nodes))
        if self.abs_idcs:
            return idcs, X_batch, y_batch, conv_matrix, center_idcs
        return X_batch, y_batch, conv_matrix, center_idcs

    @staticmethod
    def get_subgraph_information(idcs, subgraph_mat_idcs, features, labels, conv_mat_idcs,
                                 conv_mat_vals):
        num_nodes = features.shape[0]

        # bring inputs into right format
        conv_matrix = sp.csr_matrix((conv_mat_vals.numpy(),
                                     (conv_mat_idcs.numpy()[:, 0],
                                      conv_mat_idcs.numpy()[:, 1])),
                                    shape=(num_nodes, num_nodes))
        num_subgraph_idcs = subgraph_mat_idcs.shape[0]
        subgraph_mat = sp.csr_matrix((np.ones(num_subgraph_idcs, dtype=np.bool),
                                      (subgraph_mat_idcs.numpy()[:, 0],
                                       subgraph_mat_idcs.numpy()[:, 1])),
                                     shape=(num_nodes, num_nodes))

        subgraph_idcs = np.unique(subgraph_mat[idcs.numpy()].tocoo().col)
        subgraph_conv_mat = conv_matrix[subgraph_idcs, :][:, subgraph_idcs]
        temp = subgraph_conv_mat.tocoo()
        subgraph_conv_idcs = tf.constant(
            np.stack((temp.row, temp.col), axis=-1).astype(np.int64))
        subgraph_conv_vals = tf.constant(temp.data)
        subgraph_features = tf.gather(features, subgraph_idcs)
        subgraph_labels = tf.reshape(tf.gather(labels, subgraph_idcs), (-1, 1))

        # Make sure the indices of the central nodes are relative to the
        # remaining nodes in the subgraph
        idcs = np.where(np.isin(subgraph_idcs, idcs))[0]
        subgraph_labels = tf.gather(subgraph_labels, idcs)

        return (subgraph_features, subgraph_labels, subgraph_conv_idcs,
                subgraph_conv_vals, idcs, len(subgraph_idcs))

    @staticmethod
    def get_subsampled_subgraph_information(idcs, adj_list_mat, features, labels, adj_mat_idcs,
                                            sample_sizes):
        num_nodes = features.shape[0]
        num_edges = adj_mat_idcs.shape[0]

        idcs = idcs.numpy()
        sample_sizes = sample_sizes.numpy().tolist()
        adj_matrix = sp.csr_matrix((np.ones(num_edges),
                                    (adj_mat_idcs.numpy()[:, 0],
                                     adj_mat_idcs.numpy()[:, 1])),
                                   shape=(num_nodes, num_nodes),
                                   dtype=features.numpy().dtype)
        subgraph_idcs = np.unique(
            NodePredictionDataset.subsample_neighbors(idcs, sample_sizes,
                                                      adj_list_mat.numpy()))
        subgraph_adj_mat = adj_matrix[subgraph_idcs, :][:, subgraph_idcs]
        subgraph_conv_mat = normalize_adj(subgraph_adj_mat)
        temp = subgraph_conv_mat.tocoo()
        subgraph_conv_idcs = tf.constant(
            np.stack((temp.row, temp.col), axis=-1).astype(np.int64))
        subgraph_conv_vals = tf.constant(temp.data)
        subgraph_features = tf.gather(features, subgraph_idcs)
        subgraph_labels = tf.reshape(tf.gather(labels, subgraph_idcs), (-1, 1))

        # Make sure the indices of the central nodes are relative to the
        # remaining nodes in the subgraph
        idcs = np.where(np.isin(subgraph_idcs, idcs))[0]
        subgraph_labels = tf.gather(subgraph_labels, idcs)

        return (subgraph_features, subgraph_labels, subgraph_conv_idcs,
                subgraph_conv_vals, idcs, len(subgraph_idcs))

    @staticmethod
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
        if len(sample_sizes) > 1:
            new_node_idcs = np.concatenate((node_idcs, sampled_neighbors.reshape(-1)), axis=0)
            return NodePredictionDataset.subsample_neighbors(
                new_node_idcs, sample_sizes[1:], adj_list_mat)
        return np.concatenate((node_idcs, sampled_neighbors.reshape(-1)), axis=0)

    def get_adj_matrix(self):
        return sp.csr_matrix((np.ones(self.adj_idcs.shape[0]),
                              (self.adj_idcs[:, 0], self.adj_idcs[:, 1])),
                             shape=(self.num_nodes, self.num_nodes),
                             dtype=self._float_type)

    def get_corrupted_adj_information(self, rstate, adj_idcs, edge_idcs, edge_weights):
        if edge_weights is not None:
            raise NotImplementedError("Corrupting weighted graph not yet implemented")
        num_nodes = np.max(adj_idcs)+1
        directed_adj_idcs = np.array([[n1, n2] for (n1, n2) in adj_idcs if n1 < n2])
        keep_idcs = rstate.choice(directed_adj_idcs.shape[0], directed_adj_idcs.shape[0]//2, replace=False)
        existing_edges = set([(n1, n2) for n1, n2 in directed_adj_idcs[keep_idcs]])
        new_edges = set()
        while len(new_edges) != keep_idcs.shape[0]:
            missing_edges = keep_idcs.shape[0] - len(new_edges)
            random_edges = np.stack([rstate.choice(a=num_nodes, size=missing_edges, replace=True),
                                     rstate.choice(a=num_nodes, size=missing_edges, replace=True)],
                                    axis=1)
            random_edges = [(n1, n2) for (n1, n2) in random_edges if n1 < n2 and (n1, n2) not in existing_edges]
            new_edges |= set(random_edges)
        edges = list(existing_edges | new_edges)
        edges = edges + [(n2, n1) for n1, n2 in edges]
        edges = sorted(edges)
        new_adj_idcs = np.array(edges)
        new_edge_idcs = new_adj_idcs.T
        return new_adj_idcs, new_edge_idcs, None



def _get_train_idcs(nodes_per_class, labels, rstate):
    classes = sorted(np.unique(labels))
    train_idcs = set()
    for class_label in classes:
        class_idcs = np.where(labels == class_label)[0]
        class_train_idcs = set(rstate.choice(class_idcs, nodes_per_class, replace=False))
        train_idcs |= class_train_idcs
    return train_idcs


def _get_random_subset(subset_size, total_num_idcs, excluded_idcs, rstate):
    available_idcs = set(range(total_num_idcs)) - excluded_idcs
    available_idcs = np.array(list(available_idcs))
    subset_idcs = set(rstate.choice(available_idcs, subset_size, replace=False))
    return subset_idcs


def _get_random_subset_per_class(sample_per_class, labels, excluded_idcs, rstate,
                                 available_idcs=None):
    """
    :param sample_per_class: Number of data points to be selected for each
    class.
    :param labels: Labels for the whole dataset.
    :param excluded_idcs: Indices that must not be part of the resulting
    subset (e.g. because they are training or validation data).
    :param rstate: NumPy random state.
    :param available_idcs: Set of indices which the resulting subset may be
    drawn from. This may be specified instead of 'excluded_idcs'.
    :return: Set of indices of size sample_per_class * num_classes.
    """
    num_nodes = len(labels)
    if available_idcs is None:
        available_idcs = set(range(num_nodes)) - excluded_idcs
    available_idcs = np.array(list(available_idcs))
    classes = np.unique(labels)
    subset_idcs = set()
    for clss in classes:
        class_idcs = available_idcs[labels[available_idcs] == clss]
        subset_class_idcs = set(rstate.choice(class_idcs, sample_per_class,
                                              replace=False))
        subset_idcs |= subset_class_idcs
    assert len(subset_idcs) == sample_per_class * len(classes)
    assert np.all(np.bincount(labels[np.array(list(subset_idcs))]) == sample_per_class)
    return subset_idcs


def _cap_samples_per_class(train_idx, labels, training_samples_per_class, rstate):
        train_idx = set(train_idx.tolist())
        new_train_idx = _get_random_subset_per_class(
            training_samples_per_class, labels, None, rstate, train_idx)
        new_train_idx = np.array(list(new_train_idx))
        assert len(new_train_idx) == training_samples_per_class * len(np.unique(labels))
        assert np.all(np.bincount(
            labels[new_train_idx]) == training_samples_per_class)
        return new_train_idx


if __name__ == '__main__':
    # ds = NodePredictionDataset("Cora", split_seed=10,
    #                            tfidf_transform=True, standardize=True,
    #                            float_type=np.float32)
    # ds.subgraph_mat_idcs = get_subgraph_matrix_sparse(ds.adj_matrix, 2)
    # ds.get_subgraph_information(tf.constant([1024, 800, 200, 1000]),
    #                             tf.constant(ds.subgraph_mat_idcs),
    #                             tf.constant(ds.features), ds.labels,
    #                             tf.constant(ds.conv_mat_idcs),
    #                             tf.constant(ds.conv_mat_vals))

    # tl, _, _ = ds.get_data_loaders(2, 32, 32)
    # for a in tl:
    #     print("a")

    # adj = np.array([[0, 1, 0, 0],
    #                 [1, 0, 1, 1],
    #                 [0, 1, 0, 0],
    #                 [0, 1, 0, 0]])
    # print(np.matmul(adj, adj))

    ds = NodePredictionDataset("Cora", "../../data")
    adj_matrix, _, _, _, _, _, _, _ = ds.get_full_training_data()
    plot_cumulative_spectral_density(adj_matrix, "Cora")