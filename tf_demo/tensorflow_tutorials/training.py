import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

x = tf.zeros([10, 10])
x += 2
print(x)


vv = tf.Variable(1.0)
assert vv.numpy() == 1.0

# reassign the value
vv.assign(2)
assert vv.numpy() == 2.0
# use v in a tf operation like tf.square() and re-assign
vv.assign(tf.square(vv))
assert vv.numpy() == 4.0


class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(1.0)

    def __call__(self, x):
        return self.W * x + self.b


def loss(predicted_v, desired_y):
    return tf.reduce_mean(tf.square(predicted_v - desired_y))


model = Model()
assert model(3.0).numpy() == 16.0

# define a loss function
TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_B + noise

plt.figure()
plt.scatter(inputs, outputs, c="b")
plt.scatter(inputs, model(inputs), c="r")
plt.show()
print("Loss {}".format(loss(model(inputs), outputs)))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dw, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


model = Model()
ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    train(model, inputs, outputs, 0.1)

    print("Epoch {:2d}: W={:1.2f}, b={:1.2f}, loss={:2.5f}".format(epoch, model.W.numpy(), model.b.numpy(), current_loss))
plt.figure()
plt.plot(epochs, ws, 'r', epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_B] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True B'])
plt.show()

