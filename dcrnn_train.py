from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from logging import config
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()
tf.compat.v1.disable_eager_execution()
import yaml

from lib.utils import build_mask_config, load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.SafeLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        tf_config = tf.compat.v1.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=tf_config) as sess:
            mask_config = build_mask_config(adj_mx, supervisor_config.get('model', {}))
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, mask_config=mask_config, **supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
