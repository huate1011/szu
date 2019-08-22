import tensorflow as tf

tf.enable_eager_execution()

x = tf.ones([2, 3])

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz[i][j].numpy() == 12.0
del t

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_y = t.gradient(z, y)
assert dz_y.numpy() == 12.0
