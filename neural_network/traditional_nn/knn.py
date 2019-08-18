import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./", one_hot=True)
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder(dtype="float", shape=[None, 784])
xte = tf.placeholder(dtype="float", shape=[784])
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
init = tf.global_variables_initializer()
accuracy = 0.0

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("test: " + str(i) + "\t prediction: " + str(np.argmax(Ytr[nn_index])) + "\t true: " + str(np.argmax(Yte[i])))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1.0 / len(Xte)
    print("accuracy: " + str(accuracy))



