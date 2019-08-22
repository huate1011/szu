import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # tf.enable_eager_execution()
    # load imdb data with 10k words
    (train_data, train_labels), (test_data, test_labels)= keras.datasets.imdb.load_data(num_words=10000)

    # check the data shape and first review
    assert len(train_data) == len(train_labels)
    print("Training data shape {}: data[0] {}".format(train_data.shape, train_data[0]))
    assert len(test_data) == len(test_labels)
    print("Test data shape {}: data[0] {}".format(test_data.shape, test_data[0]))

    # decode review function
    word_index = {k: v+3 for k, v in keras.datasets.imdb.get_word_index().items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = word_index["<PAD>"] + 1
    word_index["<UNK>"] = word_index["<START>"] + 1
    word_index["<UNUSED"] = word_index["<UNK>"] + 1

    reverse_word_index = {v: k for k, v in word_index.items()}

    print(' '.join(reverse_word_index[x] for x in train_data[0]))

    # use keras to preprocess the train data with 256 length
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                            maxlen=256)

    # build the model, embeddig, globalaveragepooling, dense x 2
    vocal_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocal_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()

    # model compilation with adam, binary crossentropy for acc
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    # model validation
    validation_data = train_data[10000:]
    validation_labels = train_labels[10000:]

    train_data = train_data[:10000]
    train_labels = train_labels[:10000]

    # model training
    history = model.fit(train_data, train_labels, epochs=10, batch_size=512, validation_data=(validation_data, validation_labels),
                        verbose=1)

    # model evaluation
    results = model.evaluate(test_data, test_labels)

    # plot the history
    points = np.arange(1, 50+1, 1)
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.epoch, history.history['acc'], 'bo', label="train acc")
    plt.plot(history.epoch, history.history['loss'], 'b', label="train loss")
    plt.title("Train Accuracy vs Loss")
    plt.xlabel("epochs")
    plt.ylabel("results")

    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, history.history['val_acc'], 'ro', label="val acc")
    plt.plot(history.epoch, history.history['val_loss'], 'r', label="val loss")
    plt.title("Validation Accuracy vs Loss")
    plt.xlabel("epochs")
    plt.ylabel("results")

    plt.tight_layout()
    plt.legend()
    plt.show()