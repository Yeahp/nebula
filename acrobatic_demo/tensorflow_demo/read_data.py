import tensorflow as tf
import numpy as np

# method-1: repload data
a_1 = tf.constant([1.0, 2.0, 3.0])
b_1 = tf.constant([4.0, 5.0, 6.0])
c_1 = tf.add(a_1, b_1, "preload_method")

# method-2: create a placeholder for data
a_2 = tf.placeholder("float64", [1, 2])
b_2 = tf.placeholder("float64", [2, 1])
a_2_val = np.array([[1.0, 2.0], ])
b_2_val = np.array([[3.0, ], [4.0, ]])
c_2 = tf.matmul(a_2, b_2, name="placeholder_method")
d = np.array([[[1.0, 2.0, ], [3.0, 4.0, ], ], [[5.0, 6.0, ], [7.0, 8.0, ], ], [[9.0, 10.0, ], [11.0, 12.0, ], ], ])

# method-3: TFRecord
# see TFRecord.py

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    print(sess.run(c_1))

    print(sess.run(c_2, feed_dict={a_2: a_2_val, b_2: b_2_val}))


