import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULATION_RATE = 0.0001
TRAINING_STEPS = 7000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(name='weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection(name='losses', value=regularizer(weights))
        return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(shape=[INPUT_NODE, LAYER1_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(shape=[LAYER1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2


def train(mnist):
    x = tf.placeholder(dtype='float', shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(dtype='float', shape=[None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULATION_RATE)
    y = inference(x, regularizer)

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
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if (i + 1) % 1000 == 0:
                print(f"after {step} steps, the loss is {loss_value}")
                saver.save(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
