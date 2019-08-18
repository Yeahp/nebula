import tensorflow as tf

'''
here we raise an example of multi-dimension array
'''

rq = tf.RandomShuffleQueue(10, min_after_dequeue=0, dtypes=tf.int32, shapes=[1, 1, 1], name="randomQ")
init_1 = rq.enqueue_many(([[[[1], ], ], [[[2], ], ], [[[3], ], ], [[[4], ], ], [[[5], ], ]], ))
init_2 = rq.enqueue_many(([[[[6], ], ], [[[7], ], ], [[[8], ], ], [[[9], ], ], [[[10], ], ]], ))

with tf.Session() as sess:

    sess.run(init_1)
    sess.run(init_2)

    for i in range(10):
        print(sess.run(rq.dequeue())[0][0][0])

    sess.close()
