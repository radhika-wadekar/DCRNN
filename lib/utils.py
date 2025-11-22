import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf

from scipy.sparse import linalg


class DataLoader(object):

    def __init__(self, xs, ys, batch_size, tod_indices=None, dow_indices=None,
                 shuffle=True, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_with_last_sample = pad_with_last_sample
        self.xs = xs
        self.ys = ys
        self.tod_indices = tod_indices
        self.dow_indices = dow_indices

        self.use_context = tod_indices is not None and dow_indices is not None

        if self.use_context:
            self._build_context_groups()
        else:
            self.size = len(xs)
            self.num_batch = int(np.ceil(self.size / self.batch_size))

    def _build_context_groups(self):

        from collections import defaultdict
        self.context_groups = defaultdict(list)
        for i in range(len(self.xs)):
            ctx = (int(self.tod_indices[i]), int(self.dow_indices[i]))
            self.context_groups[ctx].append(i)
        self.context_groups = dict(self.context_groups)

        self.num_batch = sum(
            (len(idxs) + self.batch_size - 1) // self.batch_size
            for idxs in self.context_groups.values()
        )
        self.size = len(self.xs)

    def get_iterator(self):
        if self.use_context:
            return self._get_context_iterator()
        else:
            return self._get_standard_iterator()

    def _get_context_iterator(self):

        all_batches = []

        for (tod, dow), indices in self.context_groups.items():
            indices = np.array(indices)
            if self.shuffle:
                np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_idx = indices[start:end]

                x_batch = self.xs[batch_idx]
                y_batch = self.ys[batch_idx]

                # Pad if needed
                if self.pad_with_last_sample and len(batch_idx) < self.batch_size:
                    pad_n = self.batch_size - len(batch_idx)
                    x_batch = np.concatenate([x_batch, np.repeat(x_batch[-1:], pad_n, axis=0)])
                    y_batch = np.concatenate([y_batch, np.repeat(y_batch[-1:], pad_n, axis=0)])

                all_batches.append((x_batch, y_batch, tod, dow))

        if self.shuffle:
            np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def _get_standard_iterator(self):

        indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.size, self.batch_size):
            end = min(start + self.batch_size, self.size)
            batch_idx = indices[start:end]

            x_batch = self.xs[batch_idx]
            y_batch = self.ys[batch_idx]

            if self.pad_with_last_sample and len(batch_idx) < self.batch_size:
                pad_n = self.batch_size - len(batch_idx)
                x_batch = np.concatenate([x_batch, np.repeat(x_batch[-1:], pad_n, axis=0)])
                y_batch = np.concatenate([y_batch, np.repeat(y_batch[-1:], pad_n, axis=0)])

            yield x_batch, y_batch


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.prod([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, test_batch_size=None,
                 use_temporal_context=False, **kwargs):

    data = {}

    for cat in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, f'{cat}.npz'))
        data[f'x_{cat}'] = cat_data['x'].astype(np.float32)
        data[f'y_{cat}'] = cat_data['y'].astype(np.float32)


        if use_temporal_context:
            data[f'tod_{cat}'] = cat_data['tod']
            data[f'dow_{cat}'] = cat_data['dow']


    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(),
                            std=data['x_train'][..., 0].std())
    data['scaler'] = scaler


    for cat in ['train', 'val', 'test']:
        data[f'x_{cat}'][..., 0] = scaler.transform(data[f'x_{cat}'][..., 0])


    test_batch_size = test_batch_size or batch_size

    if use_temporal_context:
        data['train_loader'] = DataLoaderM(
            data['x_train'], data['y_train'], batch_size,
            tod_indices=data['tod_train'],
            dow_indices=data['dow_train'],
            shuffle=True
        )
        data['val_loader'] = DataLoaderM(
            data['x_val'], data['y_val'], test_batch_size,
            tod_indices=data['tod_val'],
            dow_indices=data['dow_val'],
            shuffle=False
        )
        data['test_loader'] = DataLoaderM(
            data['x_test'], data['y_test'], test_batch_size,
            tod_indices=data['tod_test'],
            dow_indices=data['dow_test'],
            shuffle=False
        )
    else:

        data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size, shuffle=True)
        data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
        data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size, shuffle=False)


    data['y_test'] = data['y_test']

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data



def build_mask_config(adj_mx, data_cfg, model_cfg):
    """
    Returns a dict with mask-related hyperparams and shapes.
    """
    use_masks = data_cfg.get('use_temporal_masks', False) or \
                model_cfg.get('use_temporal_masks', False)
    if not use_masks:
        return None

    num_nodes = adj_mx.shape[0]
    num_tod = data_cfg.get('num_tod_buckets', 4)
    num_dow = data_cfg.get('num_dow_buckets', 2)

    return {
        'use_temporal_masks': True,
        'num_nodes': num_nodes,
        'num_tod_buckets': num_tod,
        'num_dow_buckets': num_dow,
        'init_bias': 3.0   # initial raw value for all masks
    }

def calculate_random_walk_matrix_tf(adj_mx):
    """
    adj_mx: [N, N] tf.Tensor
    returns: [N, N] tf.Tensor
    """
    rowsum = tf.reduce_sum(adj_mx, axis=1, keepdims=True)  # [N, 1]
    inv_rowsum = tf.math.reciprocal_no_nan(rowsum)
    return adj_mx * inv_rowsum

def calculate_scaled_laplacian_tf(adj_mx, lambda_max=None):
    """
    TF version similar to numpy version (simplified).
    adj_mx: [N, N] tf.Tensor
    """
    # D = diag(sum A)
    d = tf.reduce_sum(adj_mx, axis=1)
    d_inv_sqrt = tf.math.reciprocal_no_nan(tf.sqrt(d))
    d_mat = tf.linalg.diag(d_inv_sqrt)
    # normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    eye = tf.eye(tf.shape(adj_mx)[0], dtype=adj_mx.dtype)
    L = eye - tf.matmul(tf.matmul(d_mat, adj_mx), d_mat)
    # scale: \tilde{L} = 2L / lambda_max - I
    if lambda_max is None:
        # crude spectral radius estimate: max diagonal
        lambda_max = tf.reduce_max(tf.linalg.eigvalsh(L))
    L_scaled = (2.0 / lambda_max) * L - eye
    return L_scaled