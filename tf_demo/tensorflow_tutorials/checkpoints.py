from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


model = create_model()
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # pass callback to training

model.save("my_model.h5")

latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)

# restore the weights
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# model reloads
model = keras.models.load_model("my_model.h5")
model.summary()
