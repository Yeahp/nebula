import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # 为此层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # 确保在函数结束时调用下面的语句
        super(MyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 这里定义了这层要实现的操作，也就是前向传播的操作
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        # 计算输出tensor的shape
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    import tensorflow.python.keras as keras
    print(keras.__version__)