import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# wget http://download.tensorflow.org/example_images/flower_photos.tgz
# wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# directory of processed images
INPUT_DATA = './path/flower_processed_data.npy'
# train path
TRAIN_PATH = './path/model'
# model path
CKPT_FILE = './path/inception_v3.ckpt'

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 32
N_CLASS = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.append(variables)


def main():
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_examples = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(inputs=images, num_classes=N_CLASS)

    trainable_variables = get_trainable_variables()
    tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, N_CLASS), logits=logits, weights=1.0)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(tf.argmax(logits, 1), labels))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # define function to load model
    load_fn = slim.assign_from_checkpoint_fn(model_path=CKPT_FILE, var_list=get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('loading model')
        sess.run(load_fn)

        start = 0
        end = BATCH
        for i in range(STEPS):
            sess.run(train_step, feed_dict={images: training_images[start:end], labels: training_labels[start:end]})
            if (i + 1) % 30 == 0 or i + 1 == STEPS:
                saver.save(sess=sess, save_path=TRAIN_PATH, global_step=i)
                validation_acc = sess.run(evaluation_step, feed_dict={images: validation_images, labels: validation_labels})
                print()
            start = end
            if start == n_training_examples:
                start = 0
            end = start + BATCH
            if end > n_training_examples:
                end = n_training_examples

        test_acc = sess.run(evaluation_step, feed_dict={images: testing_images, labels: testing_labels})
        print()


if __name__ == '__main__':
    main()
