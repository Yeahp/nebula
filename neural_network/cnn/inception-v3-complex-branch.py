import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
slim.conv2d argv: 
    inputs - input tensor, 
    num_outputs - depth of filter, 
    kernel_size - size of filter, 
    stride - step of filter, 
    padding - padding mode, 
    scope - an identity

slim.arg_scope argv:
    list_ops_or_scope - a list of op to be applied with kwargs as follows
    **kwargs - args to be applied to each member of op list above
'''

BATCH_SIZE = 100
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    net = tf.get_variable(
        name='last-layer',
        shape=[BATCH_SIZE, 32, 32, 3],
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    # name an inception module
    with tf.variable_scope('Mixed_7c'):
        # name a specific route of an inception module
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(
                inputs=net,
                num_outputs=320,
                kernel_size=[1, 1],
                stride=1,  # with arg_scope, stride can be removed
                padding='SAME',  # with arg_scope, padding can be removed
                scope='Conv2d_0a_1x1'
            )
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(
                inputs=net,
                num_outputs=320,
                kernel_size=[1, 1],
                scope='Conv2d_0a_1x1'
            )
            branch_1 = tf.concat(
                concat_dim=3,
                values=[
                    slim.conv2d(inputs=branch_1, num_outputs=384, kernel_size=[1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(inputs=branch_1, num_outputs=384, kernel_size=[3, 1], scope='Conv2d_0c_3x1')
                ]
            )
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(
                inputs=net,
                num_outputs=448,
                kernel_size=[1, 1],
                scope='Conv2d_0a_1x1'
            )
            branch_2 = slim.conv2d(
                inputs=branch_2,
                num_outputs=384,
                kernel_size=[3, 3],
                scope='Conv2d_0b_3x3'
            )
            branch_2 = tf.concat(
                concat_dim=3,
                values=[
                    slim.conv2d(inputs=branch_2, num_outputs=384, kernel_size=[1, 3], scope='Conv2d_0c_1x3'),
                    slim.conv2d(inputs=branch_2, num_outputs=384, kernel_size=[3, 1], scope='Conv2d_0d_3x1')
                ]
            )
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(
                inputs=net,
                kernel_size=[3, 3],
                scope='AvgPool_0a_3x3'
            )
            branch_3 = slim.conv2d(
                inputs=branch_3,
                num_outputs=192,
                kernel_size=[1, 1],
                scope='Conv2d_0b_1x1'
            )
        net = tf.concat(
            concat_dim=3,
            values = [
                branch_0,
                branch_1,
                branch_2,
                branch_3
            ]
        )
