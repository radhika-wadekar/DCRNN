from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import RNNCell
from lib import utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell with dynamic temporal masking."""

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=True,
                 M_tod=None, M_dow=None, tod_idx=None, dow_idx=None):
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._filter_type = filter_type

        # Store mask references for dynamic computation
        self._M_tod = M_tod
        self._M_dow = M_dow
        self._tod_idx = tod_idx
        self._dow_idx = dow_idx

        # Store original adjacency as tensor for dynamic masking
        self._A_orig = tf.convert_to_tensor(adj_mx, dtype=tf.float32)

        # For static case (no masks), precompute supports
        self._static_supports = []
        if self._M_tod is None or self._M_dow is None:
            supports = []
            if filter_type == "laplacian":
                supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
            elif filter_type == "random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            elif filter_type == "dual_random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
                supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
            else:
                supports.append(utils.calculate_scaled_laplacian(adj_mx))
            for support in supports:
                self._static_supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse.reorder(L)

    def _compute_dynamic_supports(self):
        """Compute supports dynamically based on current temporal context."""
        M_t = tf.gather(self._M_tod, self._tod_idx)  # [N, N]
        M_d = tf.gather(self._M_dow, self._dow_idx)  # [N, N]
        sig_t = tf.nn.sigmoid(M_t)
        sig_d = tf.nn.sigmoid(M_d)

        A_tilde = sig_t * sig_d * self._A_orig  # [N, N]

        supports = []
        if self._filter_type == "laplacian":
            L = self._calculate_scaled_laplacian_tf(A_tilde)
            supports.append(L)
        elif self._filter_type == "random_walk":
            rw = self._calculate_random_walk_matrix_tf(A_tilde)
            supports.append(tf.transpose(rw))
        elif self._filter_type == "dual_random_walk":
            rw_forward = self._calculate_random_walk_matrix_tf(A_tilde)
            rw_backward = self._calculate_random_walk_matrix_tf(tf.transpose(A_tilde))
            supports.append(tf.transpose(rw_forward))
            supports.append(tf.transpose(rw_backward))
        else:
            L = self._calculate_scaled_laplacian_tf(A_tilde)
            supports.append(L)

        return supports

    @staticmethod
    def _calculate_random_walk_matrix_tf(A):
        """Row-normalize adjacency: D^{-1} * A"""
        d = tf.reduce_sum(A, axis=1)  # [N]
        d_inv = tf.where(d > 0, 1.0 / d, tf.zeros_like(d))
        D_inv = tf.linalg.diag(d_inv)
        return tf.matmul(D_inv, A)

    @staticmethod
    def _calculate_scaled_laplacian_tf(A, lambda_max=None):
        """Compute scaled Laplacian: 2*L/lambda_max - I"""
        N = tf.shape(A)[0]
        d = tf.reduce_sum(A, axis=1)
        d_sqrt_inv = tf.where(d > 0, 1.0 / tf.sqrt(d), tf.zeros_like(d))
        D_sqrt_inv = tf.linalg.diag(d_sqrt_inv)

        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        I = tf.eye(N, dtype=A.dtype)
        L = I - tf.matmul(tf.matmul(D_sqrt_inv, A), D_sqrt_inv)

        if lambda_max is None:
            # Approximate lambda_max as 2 for scaled Laplacian
            lambda_max = 2.0

        # Scaled Laplacian: 2*L/lambda_max - I
        return (2.0 / lambda_max) * L - I

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """GRU with Graph Convolution."""
        with tf.compat.v1.variable_scope(scope or "dcgru_cell"):
            with tf.compat.v1.variable_scope("gates"):
                output_size = 2 * self._num_units
                if self._use_gc_for_ru:
                    fn = self._gconv
                else:
                    fn = self._fc
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
            with tf.compat.v1.variable_scope("candidate"):
                c = self._gconv(inputs, r * state, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.compat.v1.variable_scope("projection"):
                    w = tf.compat.v1.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.compat.v1.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.compat.v1.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.compat.v1.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution with DYNAMIC support computation for temporal masks."""
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        if self._M_tod is not None and self._M_dow is not None:
            supports = self._compute_dynamic_supports()  # Dense tensors [N, N]
        else:
            supports = self._static_supports  # Precomputed sparse tensors

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in supports:
                    # Handle both sparse (static) and dense (dynamic) supports
                    if isinstance(support, tf.SparseTensor):
                        x1 = tf.sparse.sparse_dense_matmul(support, x0)
                    else:
                        # Dense matmul for dynamic supports
                        x1 = tf.matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        if isinstance(support, tf.SparseTensor):
                            x2 = 2 * tf.sparse.sparse_dense_matmul(support, x1) - x0
                        else:
                            x2 = 2 * tf.matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(supports) * self._max_diffusion_step + 1
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.compat.v1.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            x = tf.matmul(x, weights)

            biases = tf.compat.v1.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.compat.v1.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)

        return tf.reshape(x, [batch_size, self._num_nodes * output_size])