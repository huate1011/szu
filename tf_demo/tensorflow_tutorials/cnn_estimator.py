from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.estimator as estimator
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """ Input function for CNN """
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # 1st cnn layer
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # 2nd cnn layer
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # 3rd cnn layer
    pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logit layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # calculate outputs and probabilities
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probablities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # return the prediction mode results
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
    return estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)

mnist_classifier = estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

tensors_log_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_log_log, every_n_iter=50)

train_input_fn = estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100,
                                                 num_epochs=None, shuffle=True)

# mnist_classifier.train(input_fn=train_input_fn, steps=1, hooks=[logging_hook])
mnist_classifier.train(input_fn=train_input_fn, steps=20000)

eval_input_fn = estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)