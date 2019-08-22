import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class RBM(object):

    def __init__(self, m, n):
        self._m = m # visible neurons
        self._n = n # hidden neurons

        self._W = tf.Variable(tf.random_normal(shape=(self._m, self._n)))
        self._c = tf.Variable(np.zeros(self._n).astype(np.float32))     # forward pass bias
        self._b = tf.Variable(np.zeros(self._m).astype(np.float32))     # backward pass bias

        self._X = tf.placeholder(np.float32, [None, self._m])

        # forward pass
        _h = tf.nn.sigmoid(tf.matmul(self._X, self._W) + self._c)
        self.h = tf.nn.relu(tf.sign(_h - tf.random_uniform(tf.shape(_h))))

        # backward pass
        _v = tf.nn.sigmoid(tf.matmul(self.h, tf.transpose(self._W)) + self._b)
        self.v = tf.nn.relu(tf.sign(_v - tf.random_uniform(tf.shape(_v))))

        # objective function
        objective = tf.reduce_mean(self.free_energy(self._X)) - tf.reduce_mean(self.free_energy(self.v))
        self._train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)

        # cross entropy cost
        reconstructed_input = self.one_pass(self._X)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._X, logits=reconstructed_input))

        self.session = None

    def fit(self, X, epochs = 10, batch_size=100):
        N, D = X.shape
        num_batches = N // batch_size
        obj = []

        for i in range(epochs):
            for j in range(num_batches):
                batch = X[j * batch_size : (j * batch_size + batch_size)]
                _, ob = self.session.run([self._train_op, self.cost], feed_dict={self._X: batch})

                if j % 10 == 0:
                    print('training epoch {0} cost {1}'.format(j, ob))
                    obj.append(ob)

        return obj

    def set_session(self, session):
        self.session = session

    def free_energy(self, v):
        b = tf.reshape(self._b, (self._m, 1))
        term_1 = -tf.matmul(v, b)
        term_1 = tf.reshape(term_1, (-1,))
        term_2 = -tf.reduce_sum(tf.nn.softplus(tf.matmul(v, self._W)) + self._c)
        return term_1 + term_2

    def one_pass(self, x):
        h = tf.nn.sigmoid(tf.matmul(x, self._W) + self._c)
        return tf.matmul(h, tf.transpose(self._W)) + self._b

    def reconstruct(self, x):
        y = tf.nn.sigmoid(self.one_pass(x))
        return self.session.run(y, feed_dict={self._X: x})


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Xtrain = trX.astype(np.float32)
    Xtest = teX.astype(np.float32)
    _, m = Xtrain.shape
    rbm = RBM(m, 100)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        rbm.set_session(sess)
        err = rbm.fit(Xtrain)
        out = rbm.reconstruct(Xtest[0:100])

    plt.plot(np.arange(len(err)) * 10, err)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.show()

    row, col = 2, 8
    idx = np.random.randint(0, 100, row * col // 2)
    f, axarr = plt.subplots(row, col, sharex=True, sharey=True, figsize=(20, 4))
    for fig, row in zip([Xtest[0:100], out], axarr):
        for i, ax in zip(idx, row):
            ax.imshow(fig[i].reshape([28, 28]), cmap="Greys_r")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
