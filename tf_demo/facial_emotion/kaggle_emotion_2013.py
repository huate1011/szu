import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

batch_size = 512
epochs = 100
data_columns = ["emotion", "pixels"]
num_classes = 7


def load_fer_data():
    fer_data = pd.read_csv(os.path.join("kaggle_emotion", "fer2013", "fer2013.csv"),
                           converters={'pixels': lambda x: np.array(x.split(' '), "float32")})
    print("number of instances: {}".format(fer_data.shape))
    return fer_data


def build_cnn(num_classes):
    model = tf.keras.models.Sequential()
    # 1st convolution layer
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    # fully connected neural networks
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def convert_train_data(train_data):
    train_x = np.asarray(train_data['pixels'].to_list())
    dims = int(np.sqrt(train_x.shape[1]))
    return train_x.reshape([train_x.shape[0], dims, dims, 1])


def plot_history(history):
    hist = pd.DataFrame(history.history)

    plt.figure()
    plt.plot(history.epoch, hist['acc'], label='accuracy')
    plt.plot(history.epoch, hist['loss'], label='loss')
    plt.xlabel("epoches")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = load_fer_data()
    train_data = dataset[dataset['Usage'] == "Training"][data_columns]
    train_x = convert_train_data(train_data)
    test_data = dataset[dataset['Usage'] == "PublicTest"][data_columns]
    test_x = convert_train_data(test_data)

    model = build_cnn(num_classes)
    # automatic fitting
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    # datagen.fit(train_x)
    # trainer = datagen.flow(train_x, train_data['emotion'].to_list(), batch_size=batch_size)
    # history = model.fit_generator(trainer, steps_per_epoch=len(train_data) / batch_size, epochs=epochs)

    # manual fitting
    # histories = []
    # for e in range(epochs):
    #     print('Epoch: {}'.format(e))
    #     batches = 0
    #     for x_batch, y_batch in datagen.flow(train_x, train_data['emotion'].to_list(), batch_size=batch_size):
    #         histories.append(model.fit(x_batch, y_batch))
    #         batches += 1
    #         if batches >= len(train_x) / batch_size:
    #             # we need to break the loop by hand because
    #             # the generator loops indefinitely
    #             break
    # plot_history(histories[0])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    history = model.fit(train_x,
                        train_data['emotion'].to_list(),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stop])
    plot_history(history)
    model.save("model_512.h5")
    model = tf.keras.models.load_model("model_512.h5")

    train_score = model.evaluate(train_x, train_data['emotion'].to_list(), verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', 100 * train_score[1])

    test_score = model.evaluate(test_x, test_data['emotion'], verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100 * test_score[1])


