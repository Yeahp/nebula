import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
IMAGE_SIZE = 28
NUM_CHANNEL = 1
NUM_LABEL = 10

# the first convolution layer
CONV1_DEEP = 8
CONV1_SIZE = 5

# the second convolution layer
CONV2_DEEP = 16
CONV2_SIZE = 5

# fully connected layer
FC_SIZE = 128

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULATION_RATE = 0.0001
TRAINING_STEPS = 7000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'


def inference(input_tensor, train, regularizer):
    '''
    :param train: distinguish train and test
    :return: raw output
    '''
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(name='weight', shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name='bias', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(name='weight', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name='bias', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped_pool2 = tf.reshape(tensor=pool2, shape=[pool_shape[0], nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(name='weight', shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # only fully connected layer is regularized
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(name='bias', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped_pool2, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(x=fc1, keep_prob=0.5)
    with tf.variable_scope('layer5-fc2'):
        fc2_weights = tf.get_variable(name='weight', shape=[FC_SIZE, NUM_LABEL], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # only fully connected layer is regularized
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(name='bias', shape=[NUM_LABEL], initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit


def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL], name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_LABEL], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULATION_RATE)
    y = inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_average_op = variable_averages.apply(var_list=tf.trainable_variables())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(inputs=tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=mnist.train.num_examples / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name="train")
    # train_op = tf.group(train_step, variable_average_op)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            _xs, _ys = mnist.train.next_batch(BATCH_SIZE)
            xs = _xs.reshape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
            ys = _ys.reshape([BATCH_SIZE, NUM_LABEL])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if (i + 1) % 1000 == 0:
                print(f"after {step} steps, the loss is {loss_value}")
                saver.save(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
