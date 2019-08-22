from tensorflow_tutorials.RBM import RBM
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RBMPredict(RBM):
    def rbm_output(self, X):
        x = tf.nn.sigmoid(tf.matmul(X, self._W) + self._c)
        return self.session.run(x, feed_dict={self._X: X})


data = pd.read_csv("kaggle13/fer2013.csv")
tr_data = data[data.Usage == "Training"]
test_data = data[data.Usage == "PublicTest"]

mask = np.random.rand(len(tr_data)) < 0.8
train_data = tr_data[mask]
val_data = tr_data[~mask]


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def preprocess_data(dataframe):
    pixels_values = dataframe.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0/255.0)

    labels_flat = dataframe['emotion'].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    labels = dense_to_one_hot(labels_flat, labels_count)
    return images, labels.astype(np.uint8)


X_train, Y_train = preprocess_data(train_data)
X_val, Y_val = preprocess_data(val_data)
X_test, Y_test = preprocess_data(test_data)


mean_image = X_train.mean(axis=0)
std_image = np.std(X_train, axis=0)
plt.imshow(mean_image.reshape(48, 48), cmap='gray')
plt.show()

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(classes)
sampels_per_class = 7

for y, cls in enumerate(classes):
    idx = np.flatnonzero(np.argmax(Y_train, axis=1) == y)
    idxs = np.random.choice(idx, sampels_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(sampels_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].reshape(48, 48), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# RBM_hidden_sizes = [1500, 700, 400]
RBM_hidden_sizes = [300]
inpX = X_train
rbm_list = []
input_size = inpX.shape[1]
for i, size in enumerate(RBM_hidden_sizes):
    print("RBM: ", i, " ", input_size, "->", size)
    rbm_list.append(RBMPredict(input_size, size))
    input_size = size

init = tf.global_variables_initializer()
for rbm in rbm_list:
    print("NEW RBM: ")
    with tf.Session() as sess:
        sess.run(init)
        rbm.set_session(sess)
        err = rbm.fit(inpX, 5)
        inpX_n = rbm.rbm_output(inpX)
        print("input shape, ", inpX_n.shape)
        inpX = inpX_n


import math
class DBN(object):
    def __init__(self, sizes, X, Y, eta=0.001, momentum=0.0, epochs=10, batch_size=100):
        self._sizes = sizes
        print(self._sizes)
        self._sizes.append(1000)
        self._X = X
        self._Y = Y
        self.N = len(X)
        self.w_list = []
        self.c_list = []
        self._learning_rate =eta
        self._momentum = momentum
        self._epochs = epochs
        self._batchsize = batch_size
        input_size = X.shape[1]

        for size in self._sizes + [Y.shape[1]]:
            max_range = 4 * math.sqrt(6. / (input_size + size))
            self.w_list.append(np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))
            self.c_list.append(np.zeros([size], np.float32))
            input_size = size

        self._a = [None] * (len(self._sizes) + 2)   #input
        self._w = [None] * (len(self._sizes) + 1)   #weights
        self._c = [None] * (len(self._sizes) + 1)   #biases

        self._a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        self.y = tf.placeholder("float", [None, self._Y.shape[1]])  #output

        for ii in range(len(self._sizes) + 1):
            self._w[ii] = tf.get_variable("W_{}".format(ii), initializer=self.w_list[ii])
            self._c[ii] = tf.get_variable("C_{}".format(ii), initializer=self.c_list[ii])

        for i in range(1, len(self._sizes) + 2):
            self._a[i] = tf.nn.sigmoid(tf.matmul(self._a[i - 1], self._w[i - 1]) + self._c[i - 1])

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y,1), logits=self._a[-1]))
        # self.cost = tf.reduce_mean(tf.square(self._a[-1] - self.y))
        self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.cost)

        self.predict_op = tf.argmax(self._a[-1], 1)

    def load_from_rbms(self, dbn_sizes, rbm_list):
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            assert dbn_sizes[i] == self._sizes[i]

        for i in range(len(self._sizes) - 1):
            self.w_list[i] = rbm_list[i]._W
            self.c_list[i] = rbm_list[i]._c

    def set_session(self, session):
        self._session = session

    def train(self, val_x, val_y):
        num_batches = self.N // self._batchsize
        batch_size = self._batchsize
        for i in range(self._epochs):
            weights = self._w
            biases = self._c
            for j in range(num_batches):
                batch = self._X[j * batch_size: (j * batch_size + batch_size)]
                batch_label = self._Y[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self._session.run([self.train_op, self.cost], feed_dict={self._a[0]: batch, self.y: batch_label})
                if j % 10 == 0:
                    print('****training epoch {0} cost {1}'.format(j, ob))
                    # print('****training w {0} c {1}'.format(self._w, ob))

                for jj in range(len(self._sizes) + 1):
                    self.w_list[jj] = self._session.run(self._w[jj])
                    self.c_list[jj] = self._session.run(self._c[jj])

            train_acc = np.mean(np.argmax(self._Y, axis=1) == self._session.run(self.predict_op, feed_dict={self._a[0]: self._X, self.y: self._Y}))
            val_acc = np.mean(np.argmax(val_y, axis=1) == self._session.run(self.predict_op, feed_dict={self._a[0]: val_x, self.y: val_y}))

            print(" epoch " + str(i) + "/" + str(self._epochs) + " training accuracy: " + str(train_acc) + " validation accuracy: " + str(val_acc))
            print("differences {} c {}".format(tf.reduce_mean(tf.square(weights[0] - self._w[0])).eval(), tf.reduce_mean(tf.square(biases[0] - self._c[0])).eval()))

    def predict(self, X):
        return self._session.run(self.predict_op, feed_dict={self._a[0]: X})


nNet = DBN(RBM_hidden_sizes, X_train, Y_train, epochs=80)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    nNet.set_session(sess)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train(X_val, Y_val)
    y_pred = nNet.predict(X_test)

