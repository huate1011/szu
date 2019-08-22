import tensorflow as tf

tf.enable_eager_execution()

layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
print(layer(tf.zeros([10, 5])))
print("layer init: {}".format(layer.get_config()))
print("layer vars: {}".format(layer.variables))
print("layer trainable vars: {}".format(layer.trainable_variables))
print("layer trainable weights: {}".format(layer.trainable_weights))
print("layer kernel: {}".format(layer.kernel))
print("layer bias: {}".format(layer.bias))


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(10)
print("My dense layer: {}".format(layer(tf.zeros([10, 5]))))
print("trainable variables: {}".format(layer.trainable_variables))


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [3,2,3])
print("***************")
print(block(tf.zeros([1,3,4,3])))
block.summary()
print([x.name for x in block.trainable_variables])