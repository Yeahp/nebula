import tensorflow as tf

q = tf.FIFOQueue(capacity=3, dtypes=tf.float32, shapes=[1, 2], name="fifo")
init = q.enqueue_many(([[[1.0, 1.0], ], [[2.0, 2.0], ], [[3.0, 3.0], ]], ))
x = q.dequeue()
y = x + [[1.0, 2.0], ]
q_inc = q.enqueue([y])

with tf.Session() as sess:

    sess.run(init)
    #sess.run(tf.global_variables_initializer())

    for i in range(3):
        sess.run(q_inc)

    print("The queue is %s with %d elements" % (q.name, sess.run(q.size())))

    for i in range(3):
        print(sess.run(x)[0][1])
