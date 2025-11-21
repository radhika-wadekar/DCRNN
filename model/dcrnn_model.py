from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, mask_config=None,**model_kwargs):
        self._scaler = scaler
        self._mask_config = mask_config

        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))


        # --- NEW: multi-context masks ---
        if mask_config is not None and mask_config.get('use_temporal_masks', False):
            num_tod = mask_config['num_tod_buckets']
            num_dow = mask_config['num_dow_buckets']
            init_bias = mask_config.get('init_bias', 3.0)

            with tf.compat.v1.variable_scope('Masks', reuse=tf.compat.v1.AUTO_REUSE):
                self.M_tod = tf.compat.v1.get_variable(
                    'M_tod',
                    shape=[num_tod, num_nodes, num_nodes],
                    initializer=tf.compat.v1.constant_initializer(init_bias),
                    trainable=True
                )
                self.M_dow = tf.compat.v1.get_variable(
                    'M_dow',
                    shape=[num_dow, num_nodes, num_nodes],
                    initializer=tf.compat.v1.constant_initializer(init_bias),
                    trainable=True
                )

            # context indices (scalar per batch)
            self.tod_idx = tf.compat.v1.placeholder(tf.int32, shape=(), name='tod_idx')
            self.dow_idx = tf.compat.v1.placeholder(tf.int32, shape=(), name='dow_idx')
        else:
            self.M_tod = None
            self.M_dow = None
            self.tod_idx = None
            self.dow_idx = None


        self._inputs = tf.compat.v1.placeholder(
            tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs'
        )
        self._labels = tf.compat.v1.placeholder(
            tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels'
        )

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

        cell = DCGRUCell(
            rnn_units,
            adj_mx,
            max_diffusion_step=max_diffusion_step,
            num_nodes=num_nodes,
            filter_type=filter_type,
            M_tod=self.M_tod,
            M_dow=self.M_dow,
            tod_idx=self.tod_idx,
            dow_idx=self.dow_idx
        )
        cell_with_projection = DCGRUCell(
            rnn_units,
            adj_mx,
            max_diffusion_step=max_diffusion_step,
            num_nodes=num_nodes,
            num_proj=output_dim,
            filter_type=filter_type,
            M_tod=self.M_tod,
            M_dow=self.M_dow,
            tod_idx=self.tod_idx,
            dow_idx=self.dow_idx
        )

        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(decoding_cells, state_is_tuple=True)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v1.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(
                tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1
            )
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1
            )
            labels.insert(0, GO_SYMBOL)

            def _loop_function(prev, i):
                if is_training:
                    if use_curriculum_learning:
                        c = tf.random.uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    result = prev
                return result

            _, enc_state = tf.compat.v1.nn.static_rnn(encoding_cells, inputs, dtype=tf.float32)


            outputs = []
            state = enc_state
            prev = labels[0]

            with tf.compat.v1.variable_scope('rnn_decoder'):
                for i in range(1, len(labels)):
                    current_input = _loop_function(prev, i)
                    output, state = decoding_cells(current_input, state)
                    outputs.append(output)
                    prev = output


        outputs = tf.stack(outputs, axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')


        self._merged = tf.compat.v1.summary.merge_all()


    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """Inverse sigmoid decay for scheduled sampling."""
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)


    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs
