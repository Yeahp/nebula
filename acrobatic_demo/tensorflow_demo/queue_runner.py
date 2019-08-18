import tensorflow as tf

q = tf.FIFOQueue(1000, "int32", name="a_fifo_queue")
counter = tf.Variable(0)
increment_op = tf.assign_add(counter, tf.constant(1))
enqueue_op = q.enqueue([counter])
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess,
                                    coord=coord,
                                    start=True)
for i in range(10):
    print(sess.run(q.dequeue()))

coord.request_stop()

for i in range(5):
    try:
        print(sess.run(q.dequeue()))
    except tf.errors.OutOfRangeError:
        break
coord.join(enqueue_threads)

sess.close()
