import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


class TF_PCA:
    def __init__(self, data, dtype=tf.float32):
        self._data = data
        self._dtype = dtype
        self._graph = None
        self._X = None
        self._u = None
        self._singular_values = None
        self._sigma = None

    def fit(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._X = tf.placeholder(self._dtype, shape=self._data.shape)
            # Perform SVD
            singular_values, u, _ = tf.svd(self._X)
            sigma = tf.diag(singular_values)
        with tf.Session(graph=self._graph) as session:
            self._u, self._singular_values, self._sigma = session.run([u, singular_values, sigma],
                                                                      feed_dict={self._X: self._data})

    def reduce(self, n_dimensions=None, keep_info=None):
        if keep_info:
            normalized_singular_values = self._singular_values / sum(self._singular_values)
            info = np.cumsum(normalized_singular_values)
            it = iter(idx for idx, value in enumerate(info) if value >= keep_info)
            n_dimensions = next(it) + 1
        with self._graph.as_default():
            sigma = tf.slice(self._sigma, [0, 0], [self._data.shape[1], n_dimensions])
            pca = tf.matmul(self._u, sigma)
        with tf.Session(graph=self._graph) as session:
            return session.run(pca, feed_dict={self._X: self._data})


tf_pca = TF_PCA(mnist.train.images)
tf_pca.fit()
pca = tf_pca.reduce(keep_info=0.1)
print('original data shape', mnist.train.images.shape)
print('reduced data shape', pca.shape)

sns_set = sns.color_palette("Set2", 10)
color_mapping = {key: value for (key, value) in enumerate(sns_set)}
colors = list(map(lambda x: color_mapping[x], mnist.train.labels))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=colors)
fig.show()

## tf embedding
import os
from tensorflow.contrib.tensorboard.plugins import projector
LOG_DIR = "tf_mnist"
images = tf.Variable(mnist.test.images, name='images')
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
with open(metadata, 'w') as metadata_file:
    for row in mnist.test.labels:
        metadata_file.write('%d\n' % row)

with tf.Session() as sess:
    saver = tf.train.Saver([images])
    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
