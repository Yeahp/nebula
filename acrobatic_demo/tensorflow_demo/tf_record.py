import tensorflow as tf
import numpy as np

tfrecords_filename = '/Users/hello/Desktop/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        img_raw = np.random.randint(0, 255, size=(1, 7))
        img_raw = img_raw.tostring()
        print(img_raw)
        #example = tf.train.Example(features=tf.train.Features(
        #    feature={
        #        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        #        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        #    }))
        #writer.write(example.SerializeToString())
    #writer.close()
    sess.close()
