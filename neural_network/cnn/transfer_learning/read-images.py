import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# directory of input images
INPUT_DATA = './path/flower_photo'
# directory of output images stored in numpy
OUTPUT_FILE = './path/flower_processed_data.npy'

# proportion of validation set
VALIDATION_PERCENTAGE = 10
# proportion of test set
TEST_PERCENTAGE = 10


def create_images_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(sub_dir, '*.' + extension)
            for file in glob.glob(file_glob):
                file_list.append(file)
        if not file_list:
            continue

        # process images
        for file_name in file_list: # 299 * 299
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(contents=image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
            image = tf.image.resize_images(images=image, size=[299, 299])
            image_value = sess.run(image)

            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    # shuffle data
    state = np.random.get_state()
    np.random.shuffle(testing_images)
    np.random.set_state(state)
    np.random.shuffle(testing_labels)

    return np.asarray([training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_images_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()
