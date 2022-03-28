import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.model.vae import SCPHERE
from scphere_2.util.trainer import Trainer
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

random_tool = tf.compat.v1.keras.utils.DeterministicRandomTestTool()

with random_tool.scope():
  graph = tf.Graph()
  with graph.as_default(), tf.compat.v1.Session(graph=graph) as sess:
    mtx = '/home/users/nlaabid/sc-Hemap/experiments/scPhere/example/data/cd14_monocyte_erythroid.mtx'
    x = read_mtx(mtx, dtype='float32')
    x = x.transpose().todense()
    x_tf = tf.convert_to_tensor(x)

    batch = np.zeros(x.shape[0]) * -1

    model = SCPHERE(x=x, batch=batch, n_gene=x.shape[1], n_batch=0, batch_invariant=True,
                z_dim=2, latent_dist='normal',
                observation_dist='nb', seed=0)

    sess.run(tf.compat.v1.global_variables_initializer())

    init_vars = []
    for var in tf.compat.v1.global_variables():
      init_vars.append(sess.run(var))

    max_epoch = 5
    trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=max_epoch, init_vars=init_vars,
                      mb_size=x.shape[0], learning_rate=0.001)
    trainer.train()

    feed_dict = {model.x: x, model.batch_id: batch}

    tf1_elbo = sess.run(model.ELBO, feed_dict=feed_dict)
    tf1_kl = sess.run(model.kl, feed_dict=feed_dict)
    tf1_log_likelihood = sess.run(model.log_likelihood, feed_dict=feed_dict)
    tf1_depth_loss = sess.run(model.depth_loss, feed_dict=feed_dict)


print("tf1_depth_loss: ", tf1_depth_loss)
print("tf1_elbo: ", tf1_elbo)
print("tf1_log_likelihood: ", tf1_log_likelihood)
print("tf1_kl: ", tf1_kl)