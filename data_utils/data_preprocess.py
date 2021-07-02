import cv2
import tensorflow as tf
from glob import glob
import numpy as np
from data_utils.utils import check_img_label_list, shuffle_file, check_file_is_aug


def get_img_mask_list(file_path, batch_size, file_number=0, data_augmentation=0):
    """
    将图像和标签数据队列处理后以tensor返回
    图像格式为(size, size, 3)
    标签格式为(size, size)

    :param batch_size:
    :param data_augmentation:
    :param file_number:可以截取一部分数据
    :param file_path:
    :return:
    """

    img_path = file_path + 'img/'
    label_path = file_path + 'label/'

    img_file_path_list = glob(img_path + '*.jpg')
    label_file_path_list = glob(label_path + '*.png')
    assert len(img_file_path_list) == len(label_file_path_list)
    print('[INFO] 数量校验通过')
    """
    正序保证文件对应
    下面的check_img_label_list函数也是为了保证文件对应
    对应后会匹配成对再打乱
    不过目前的数据集打不打乱都可以
    """
    img_file_path_list.sort()
    label_file_path_list.sort()

    """
    对list内文件的文件名进行判断  标准数据的命名不带'_'  增强数据含'_'  
    eg： img：123.jpg  label：123.png      aug_img：123_nose.jpg  aug_label：123_nose.png
    而后根据data_augmentation的比例 向标准数据队列中注入一定比例的增强数据
    """
    img_file_path_list, label_file_path_list, aug_img_file_path_list, aug_label_file_path_list \
        = check_file_is_aug(img_file_path_list, label_file_path_list)
    if 0 < data_augmentation <= 1:
        aug_img_file_path_list, aug_label_file_path_list = shuffle_file(aug_img_file_path_list, aug_label_file_path_list)
        img_file_path_list.extend(aug_img_file_path_list[:int(data_augmentation*len(aug_img_file_path_list))])
        label_file_path_list.extend(aug_label_file_path_list[:int(data_augmentation * len(aug_label_file_path_list))])

    # 截取部分文件
    if file_number > 0:
        print('[INFO] 截取部分文件 其数量为：\t' + str(file_number))
        if file_number > len(img_file_path_list):
            file_number = len(img_file_path_list)
        img_file_path_list, label_file_path_list = shuffle_file(img_file_path_list, label_file_path_list)
        img_file_path_list = img_file_path_list[:file_number]
        label_file_path_list = label_file_path_list[:file_number]
    else:
        print('[INFO] 不截取文件 其数量为：\t' + str(len(img_file_path_list)))

    check_img_label_list(img_file_path_list, label_file_path_list)

    image_label_ds = tf.data.Dataset.from_tensor_slices((img_file_path_list, label_file_path_list))
    # image_label_ds = image_label_ds.map(load_and_preprocess_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    image_label_ds = image_label_ds.map(load_and_preprocess_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    # image_label_ds = image_label_ds.cache()
    image_label_ds = image_label_ds.shuffle(buffer_size=batch_size * 8)
    image_label_ds = image_label_ds.batch(batch_size=batch_size)
    image_label_ds = image_label_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return image_label_ds


def load_and_preprocess_image_label(img_path, label_path):
    """
    对img和label进行读取预处理

    :param label_path:
    :param img_path:
    :return:
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    # label = tf.image.resize(label, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # label = tf.reshape(tensor=label, shape=(512, 512))
    label = tf.cast(label, tf.float32) / 255.0

    return image, label


def load_and_preprocess_image_onehot_label(img_path, label_path):
    """
    对img和label进行读取预处理 其中label以onehot的形式

    :param label_path:
    :param img_path:
    :return:
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.reshape(tensor=label, shape=(512, 512))

    label = tf.cast(label, dtype=tf.uint8)
    label = tf.one_hot(indices=label, depth=20)

    return image, label
