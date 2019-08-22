import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(' ')
        print('.', end='')


def normal_data(x):
    return (x - train_stats['mean']) / train_stats['std']


def build_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[input_shape]))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))
    model.compile(metrics=['mean_absolute_error', 'mean_squared_error'],
                  loss="mean_squared_error",
                  optimizer=tf.keras.optimizers.RMSprop()
                  )
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist.tail()

    plt.clf()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.epoch, hist['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.epoch, hist['val_mean_absolute_error'], label='val_mean_absolute_error')
    plt.ylim([0, 5])
    plt.xlabel("epoches")
    plt.ylabel("Absolute error")

    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, hist['mean_squared_error'], label='mean_squared_error')
    plt.plot(history.epoch, hist['val_mean_squared_error'], label='val_mean_squared_error')
    plt.ylim([0, 20])
    plt.xlabel("epoches")
    plt.ylabel("Square error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # get the dataset

    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)
    with open(dataset_path,"r") as data_file:
        print(data_file.readline())
        print(data_file.readline())

    # import the data
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)
    print(raw_dataset.describe([0.1]))

    # clean the dataset
    dataset = raw_dataset.dropna()
    dataset.isna().sum()
    assert np.sum(dataset.isna().sum()) == 0

    # split the dataset into training and testing
    train_dataset = dataset.sample(frac=0.2, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # seaborn plot training data
    plt.figure()
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.show()

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")

    # normalize the data
    normal_train_data = normal_data(train_dataset)
    normal_test_data = normal_data(test_dataset)

    # build model
    model = build_model(len(test_dataset.keys()))
    model.summary()

    # model prediction
    prediction_results = model.predict(normal_train_data[:10])
    plt.clf()
    plt.plot(np.arange(0, len(prediction_results)), prediction_results, label="prediction")
    plt.title("Prediction results")
    plt.xlabel("epoches")
    plt.ylabel("prediction results")

    plt.plot(np.arange(1, len(prediction_results) + 1), prediction_results, label="prediction")
    plt.title("Prediction results")
    plt.xlabel("epoches")
    plt.ylabel("prediction results")
    plt.legend()
    plt.show()

    # train the model
    EPOCHS = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(normal_train_data, train_labels, batch_size=10, epochs=EPOCHS, verbose=0, validation_split=0.2,
                        callbacks=[PrintDot(), early_stop])
    plot_history(history)

    # model evaluation
    loss, mae, mse = model.evaluate(normal_test_data, test_labels, batch_size=10)
    print("Testing set Mean Abs Error: {:5.2f} MPG\n"
          "Testing set Mean squared Error: {:5.2f} MPG\n"
          "Testing loss: {:5.2f} MPG".format(mae, mse, loss))

    # model prediction
    test_predictions = model.predict(normal_test_data).flatten()
    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True values[MPG]")
    plt.ylabel("Predictions[MPG]")
    plt.axis('equal')
    plt.axis('square')
    # plt.xlim([0, plt.xlim()[1]])
    # plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    # histgram
    plt.figure()
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction error [mpg]")
    plt.ylabel("Count")
    plt.show()