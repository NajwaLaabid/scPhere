
import pandas as pd
import numpy as np

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.util.trainer import Trainer
from scphere_2.model.vae import SCPHERE
from scphere_2.util.plot import plot_trace
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

# %%
data_dir = '../data/'
# mtx = data_dir + 'cd14_monocyte_erythroid.mtx'
mtx = '/home/users/nlaabid/sc-Hemap/experiments/scPhere/example/data/cd14_monocyte_erythroid.mtx'
x = read_mtx(mtx, dtype='float32')
x = x.transpose().todense()

batch = np.zeros(x.shape[0]) * -1

model = SCPHERE(x=x, batch=batch, n_gene=x.shape[1], n_batch=0, batch_invariant=True,
                z_dim=2, latent_dist='normal',
                observation_dist='nb', seed=0)

# %%
max_epoch = 10
trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=max_epoch,
                  mb_size=x.shape[-1], learning_rate=0.001)

trainer.train()



