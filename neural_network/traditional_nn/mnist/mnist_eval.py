import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import neural_network.traditional_nn.mnist.mnist_train as mnist_train

EVAL_INTERVAL_SECS = 5


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_train.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_train.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_train.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(decay=mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session(graph=g) as sess:
            while True:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)  # check the latest model
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_acore = sess.run(accuracy, feed_dict=validate_feed)
                    print(f"after {global_step} steps, the accuracy is {accuracy_acore}")
                else:
                    print('no checkpoint file found!')
                    break
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
