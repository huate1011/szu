import os
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
features = columns[:-1]
labels = columns[-1]

class_name = ["Iris setosa", "Iris versicolor", "Iris virginica"]

batch_size = 32
train_dataset = tf.contrib.data.make_csv_dataset(train_dataset_fp, batch_size, column_names=columns, label_name=labels, num_epochs=1)
features, labels = next(iter(train_dataset))
print(features)

plt.figure()
plt.scatter(features["sepal_length"].numpy(), labels.numpy(), c=labels.numpy())
plt.xlabel("sepal length")
plt.ylabel("labels")
plt.show()


def pack_features_labels(features, labels):
    return tf.stack(list(features.values()), axis=1), labels


train_dataset = train_dataset.map(pack_features_labels)
features, labels = next(iter(train_dataset))
print(features[:5])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=[features.shape[1]]),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

predictions = model(features)
print(predictions[:5])
print(tf.nn.softmax(predictions[:5]))

print("Predictions: {}".format(tf.argmax(predictions, axis=1)))
print("labels: {}".format(labels))


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.GradientDescentOptimizer(0.01)
global_step = tf.Variable(0)
loss_value, gradient = grad(model, features, labels)
print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(gradient, model.trainable_variables), global_step)
print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss(model, features, labels)))

tfe = tf.contrib.eager
train_loss_results = []
train_accuracy_results = []

num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))


fig, axes = plt.subplots(2, sharex=True, figsize=[12,8])
fig.suptitle("training metrics")

axes[0].set_ylabel("Loss")
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Epoches")
axes[1].plot(range(num_epochs), train_accuracy_results)
fig.show()