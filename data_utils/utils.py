# coding=utf-8
import tensorflow as tf
from glob import glob
import numpy as np
import os
import shutil


def get_img_mask_list(file_path, batch_size, file_number=0, data_augmentation=False):
    """
    将图像和标签数据队列处理后以tensor返回
    图像格式为(size, size, 3)
    标签格式为(size, size, 1)
    标签总计2类(含背景)
    :param batch_size:
    :param data_augmentation:
    :param file_number:可以节取一部分数据
    :param file_path:
    :return:
    """
    autotune = tf.data.experimental.AUTOTUNE

    img_path = file_path + 'img/'
    label_path = file_path + 'label/'

    if data_augmentation:
        print('调用数据增强后的文件')
        img_path = file_path + 'aug_img/'
        label_path = file_path + 'aug_label/'

    img_file_path_list = glob(img_path + '*.jpg')
    label_file_path_list = glob(label_path + '*.png')
    """
    正序保证文件对应
    下面的check_img_label_list函数也是为了保证文件对应
    对应后会匹配成对再打乱
    不过目前的数据集打不打乱都可以
    """
    img_file_path_list.sort()
    label_file_path_list.sort()
    assert len(img_file_path_list) == len(label_file_path_list)

    # 截取部分文件
    if file_number > 0:
        print('截取部分文件 其数量为：\t' + str(file_number))
        if file_number > len(img_file_path_list):
            file_number = len(img_file_path_list)
        img_file_path_list = img_file_path_list[:file_number],
        label_file_path_list = label_file_path_list[:file_number]
    else:
        print('不截取文件 其数量为：\t' + str(len(img_file_path_list)))

    check_img_label_list(img_file_list=img_file_path_list, label_file_list=label_file_path_list)

    image_label_ds = tf.data.Dataset.from_tensor_slices((img_file_path_list, label_file_path_list))
    image_label_ds = image_label_ds.map(load_and_preprocess_image_label, num_parallel_calls=autotune)

    image_label_ds = image_label_ds.shuffle(buffer_size=batch_size * 4)
    image_label_ds = image_label_ds.batch(batch_size)
    image_label_ds = image_label_ds.prefetch(buffer_size=autotune)

    return image_label_ds


def load_and_preprocess_image_label(img_path, label_path):
    """
    对img和label进行预处理
    需要注意的是为了保持label的准确其实不该做resize的 我是没有想到更好的方法才做的resize
    这里因为用到了bn，会要求同一batch的size相同，所以用最近邻保持标签数值
    :param label_path:
    :param img_path:
    :return:
    """
    print(img_path, label_path)

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.reshape(tensor=label, shape=(512, 512))
    label = label // 255
    label = tf.cast(label, tf.uint8)
    label = tf.one_hot(indices=label, depth=2, on_value=1, off_value=0)

    print(image.shape, label.shape)
    return image, label


def shuffle_file(img_file_list, label_file_list):
    """
    打乱img和label的文件列表顺序 并返回两列表 seed已固定
    :param img_file_list:
    :param label_file_list:
    :return:
    """
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    return img_file_list, label_file_list


def distribution_file(dis_img_file_list,
                      dis_label_file_list,
                      dis_img_file_path,
                      dis_label_file_path):
    """
    将img和label从一文件夹转至其他位置
    :param dis_img_file_list:
    :param dis_label_file_list:
    :param dis_img_file_path:
    :param dis_label_file_path:
    :return:
    """
    recreate_dir(dis_img_file_path)
    recreate_dir(dis_label_file_path)
    for img_file, label_file in zip(dis_img_file_list, dis_label_file_list):
        img_name = img_file.split('/')[-1]
        label_name = label_file.split('/')[-1]

        shutil.copyfile(img_file, dis_img_file_path + img_name)
        shutil.copyfile(label_file, dis_label_file_path + label_name)
        # print(img_file, label_file, dis_img_file_path + img_name, dis_label_file_path + label_name)


def create_dir(folder_name):
    """
    创建文件夹
    :param folder_name:
    :return:
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def recreate_dir(folder_name):
    """
    重建文件夹
    :param folder_name:
    :return:
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        shutil.rmtree(folder_name)
        create_dir(folder_name)


def check_img_label_list(img_file_list, label_file_list):
    for img_path, label_path in zip(img_file_list, label_file_list):
        img_name = (img_path.split('/')[-1]).split('.')[0]
        label_name = (label_path.split('/')[-1]).split('.')[0]

        assert img_name == label_name
    print('文件对应检查通过')
