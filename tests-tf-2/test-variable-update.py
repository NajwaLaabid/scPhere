from contextlib import contextmanager
import tensorflow as tf


tf.compat.v1.disable_v2_behavior()

@contextmanager
def assert_no_variable_creations():
  """Assert no variables are created in this context manager scope."""
  def invalid_variable_creator(next_creator, **kwargs):
    raise ValueError("Attempted to create a new variable instead of reusing an existing one. Args: {}".format(kwargs))

  with tf.variable_creator_scope(invalid_variable_creator):
    yield

@contextmanager
def catch_and_raise_created_variables():
  """Raise all variables created within this context manager scope (if any)."""
  created_vars = []
  def variable_catcher(next_creator, **kwargs):
    var = next_creator(**kwargs)
    created_vars.append(var)
    return var

  with tf.variable_creator_scope(variable_catcher):
    yield
  if created_vars:
    raise ValueError("Created vars:", created_vars)


import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.util.trainer import Trainer
from scphere_2.model.vae import SCPHERE
from scphere_2.util.plot import plot_trace
import numpy as np

# %%
data_dir = '../data/'
# mtx = data_dir + 'cd14_monocyte_erythroid.mtx'
mtx = '/home/users/nlaabid/sc-Hemap/experiments/scPhere/example/data/cd14_monocyte_erythroid.mtx'
x = read_mtx(mtx)
x = x.transpose().todense()

batch = np.zeros(x.shape[0]) * -1

model = SCPHERE(n_gene=x.shape[1], n_batch=0, batch_invariant=True,
                z_dim=2, latent_dist='normal',
                observation_dist='nb', seed=0)

max_epoch = 10
trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=max_epoch,
                  mb_size=x.shape[-1], learning_rate=0.001)

# Verify that no new weights are created in followup calls
with assert_no_variable_creations():
  trainer.train()
with catch_and_raise_created_variables():
  trainer.train()