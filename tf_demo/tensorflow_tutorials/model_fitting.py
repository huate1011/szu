from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def multi_hot_seq(sequences, dimension):
    results = np.zeros([len(sequences), dimension])
    for i, index in enumerate(sequences):
        results[i, index] = 1.0
    return results


def plot_history(histories, key: str = 'binary_crossentropy'):
    plt.figure(figsize=[16, 10])
    for name, history in histories.items():
        plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + " Val")
        plt.plot(history.epoch, history.history[key], label=name.title() + " Train")
    plt.xlabel("epochs")
    plt.ylabel(key.title())
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_words = 10000
    (training_data, training_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=num_words)
    training_data = multi_hot_seq(training_data, num_words)
    test_data = multi_hot_seq(test_data, num_words)

    plt.figure()
    plt.plot(training_data[0])
    plt.show()

    # create baseline model
    base_model = keras.models.Sequential()
    base_model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=[num_words, ]))
    base_model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    base_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    base_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", "binary_crossentropy"])
    base_model.summary()

    # create smaller model
    small_model = keras.models.Sequential()
    small_model.add(keras.layers.Dense(4, activation=tf.nn.relu, input_shape=[num_words, ]))
    small_model.add(keras.layers.Dense(4, activation=tf.nn.relu))
    small_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    small_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", "binary_crossentropy"])
    small_model.summary()

    # create bigger model
    big_model = keras.models.Sequential()
    big_model.add(keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[num_words, ]))
    big_model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    big_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    big_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", "binary_crossentropy"])
    big_model.summary()

    # fit models
    base_hist = base_model.fit(training_data, training_labels, batch_size=512, epochs=20, verbose=2,
                               validation_data=[test_data, test_labels])
    small_hist = small_model.fit(training_data, training_labels, batch_size=512, epochs=20, verbose=2,
                                 validation_data=[test_data, test_labels])
    big_hist = big_model.fit(training_data, training_labels, batch_size=512, epochs=20, verbose=2,
                             validation_data=[test_data, test_labels])

    plot_history({'base model': base_hist, 'small hist': small_hist, 'big hist': big_hist})

    # l2 regularization
    l2_model = keras.models.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu, input_shape=[num_words, ]),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    l2_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", "binary_crossentropy"])
    l2_hist = l2_model.fit(training_data, training_labels, batch_size=512, epochs=20, verbose=2,
                           validation_data=[test_data, test_labels])

    plot_history({'base model': base_hist, 'l2 hist': l2_hist})

    # drop out
    dpt_model = keras.models.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(num_words,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    dpt_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

    dpt_model_history = dpt_model.fit(training_data, training_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
