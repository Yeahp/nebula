import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./', one_hot=True)
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

X = tf.placeholder(dtype="float", shape=[None, num_input])
Y = tf.placeholder(dtype=float, shape=[None, num_classes])

weights = {
    'h1': tf.Variable(initial_value=tf.random_normal(shape=[num_input, n_hidden_1])),
    'h2': tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_1, n_hidden_2])),
    'out': tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_1])),
    'b2': tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_2])),
    'out': tf.Variable(initial_value=tf.random_normal(shape=[num_classes]))
}

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# construct model
logits = neural_net(X)

# define loss and optimizer
loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# evaluate model (with test logits, for dropout to be disabled)
is_correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_pred, tf.float32))

init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    sess.run(init)

    for step in range(num_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # run optimization op (backprop)
        sess.run(fetches=train_op, feed_dict={X: batch_x, Y: batch_y})

        if (step + 1)%display_step == 0 or step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("step " + str(step + 1) + "\t batch loss: " + str(loss) + "\t accuracy: " + str(acc))

    # calculate accuracy for test
    print("test accuracy: " + str(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))




