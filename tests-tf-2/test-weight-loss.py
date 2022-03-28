import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.model.vae import SCPHERE
from scphere_2.util.trainer import Trainer

import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

graph = tf.Graph()
with graph.as_default(), tf.compat.v1.Session(graph=graph) as sess:
  mtx = '/home/users/nlaabid/sc-Hemap/experiments/scPhere/example/data/cd14_monocyte_erythroid.mtx'
  x = read_mtx(mtx, dtype='float32')
  x = x.transpose().todense()

  x_tf = tf.convert_to_tensor(x)
  x_ds = tf.data.Dataset.from_tensors(x_tf)

  batch = np.zeros(x.shape[0]) * -1

  model = SCPHERE(x=x, batch=batch, n_gene=x.shape[1], n_batch=0, batch_invariant=True,
                z_dim=2, latent_dist='normal',
                observation_dist='nb', seed=0)

  max_epoch = 2
  trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=max_epoch,
                  mb_size=x.shape[0], learning_rate=0.001)
  trainer.train()


  feed_dict = {model.x: x, model.batch_id: batch}

  # Rather than running the global variable initializers,
  # reset all variables to a constant value
  l = []
  for var in tf.compat.v1.global_variables():
      if var.dtype == 'int64':
          mul = 1
      else:
          mul = 0.001
      l.append(var.assign(tf.ones_like(var) * mul))

  var_reset = tf.group(l)
  sess.run(var_reset)

  # # Grab the outputs & regularization loss

  # elbo = tf.compat.v1.get_collection('elbo')
  # kl = tf.compat.v1.get_collection('kl')
  # log_likelihood = tf.compat.v1.get_collection('log_likelihood')
  # depth_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

  tf1_elbo = sess.run(model.ELBO, feed_dict=feed_dict)
  tf1_kl = sess.run(model.kl, feed_dict=feed_dict)
  tf1_log_likelihood = sess.run(model.log_likelihood, feed_dict=feed_dict)
  tf1_depth_loss = sess.run(model.depth_loss, feed_dict=feed_dict)

  # mu, sigma_square = sess.run([model.mu, model.sigma_square], feed_dict=feed_dict)

print("tf1_depth_loss: ", tf1_depth_loss)
print("tf1_elbo: ", tf1_elbo)
print("tf1_log_likelihood: ", tf1_log_likelihood)
print("tf1_kl: ", tf1_kl)



# print("Mu: ", mu)
# print("Sigma_square: ", sigma_square)
#tf1_output[0][:5]