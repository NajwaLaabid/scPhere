import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.model.vae import SCPHERE
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

graph = tf.Graph()
with graph.as_default(), tf.compat.v1.Session(graph=graph) as sess:
    mtx = '/home/users/nlaabid/sc-Hemap/experiments/scPhere/example/data/cd14_monocyte_erythroid.mtx'
    x = read_mtx(mtx, dtype='float32')
    x = x.transpose().todense()

    batch = np.zeros(x.shape[0]) * -1

    model = SCPHERE(n_gene=x.shape[1], n_batch=0, batch_invariant=True,
                z_dim=2, latent_dist='normal',
                observation_dist='nb', seed=0)

    tf1_variable_names_and_shapes = {
        var.name: (var.trainable, var.shape) for var in tf.compat.v1.global_variables()}
    num_tf1_variables = len(tf.compat.v1.global_variables())


print('num_tf1_variables ', num_tf1_variables)
print('tf1_variable_names_and_shapes ', tf1_variable_names_and_shapes)
