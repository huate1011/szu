import tensorflow as tf
import numpy as np
import time


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, tf.matrix_transpose(x))
    end = time.time() - start
    print("10 loops: {:0.2f} ms".format(end))


if __name__ == "__main__":
    tf.enable_eager_execution()
    print(tf.add(1, 2))
    print(tf.add(np.array([1,2]), np.array([3,4])))
    print(tf.sqrt(4.0))
    print(tf.squared_difference(4, 8))
    print(tf.reduce_sum(np.array([1,2,3])))
    print(tf.square(6) + tf.cast(tf.sqrt(8.0), dtype=tf.int32))
    print(tf.encode_base64("hello world"))
    print(tf.matmul([[2],[3]], [[2,3]]))

    # test numpy convert
    print(tf.multiply(np.array(np.ones((3,4))), 2))
    print(np.add(tf.sqrt(4.0), 1))
    print(tf.sqrt(4.0).numpy())

    # test gpu
    x = tf.random_normal([3,3])
    print("Is there a gpu : {bg}".format(bg=tf.test.is_gpu_available()))
    print("is the tensor on gpu #0: {}".format(x.device))

    # timing
    with tf.device("CPU:0"):
        x = tf.random_normal([3, 4])
        assert x.device.endswith("CPU:0")
        time_matmul(x)

    