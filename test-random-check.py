import sys
sys.path.insert(0, './')

from scphere_2.util.util import read_mtx
from scphere_2.model.vae import SCPHERE
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

random_tool = tf.compat.v1.keras.utils.DeterministicRandomTestTool()

with random_tool.scope():
  graph = tf.Graph()
  with graph.as_default(), tf.compat.v1.Session(graph=graph) as sess:
    a = tf.random.uniform(shape=(3,1))
    a = a * 3
    b = tf.random.uniform(shape=(3,3))
    b = b * 3
    c = tf.random.uniform(shape=(3,3))
    c = c * 3
    graph_a, graph_b, graph_c = sess.run([a, b, c])

print('graph_a ', graph_a)
print('graph_b ', graph_b)
print('graph_c ', graph_c)