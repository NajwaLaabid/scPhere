import tensorflow as tf

random_tool = tf.compat.v1.keras.utils.DeterministicRandomTestTool()
with random_tool.scope():
  a = tf.random.uniform(shape=(3,1))
  a = a * 3
  b = tf.random.uniform(shape=(3,3))
  b = b * 3
  c = tf.random.uniform(shape=(3,3))
  c = c * 3

print('a ', a)
print('b ', b)
print('c ', c)