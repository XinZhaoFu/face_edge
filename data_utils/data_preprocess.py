import tensorflow as tf
from glob import glob
from data_utils.utils import check_img_label_list


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
    assert len(img_file_path_list) == len(label_file_path_list)
    print('[info] 数量校验通过')
    """
    正序保证文件对应
    下面的check_img_label_list函数也是为了保证文件对应
    对应后会匹配成对再打乱
    不过目前的数据集打不打乱都可以
    """
    img_file_path_list.sort()
    label_file_path_list.sort()

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
    对img和label进行读取预处理
    :param label_path:
    :param img_path:
    :return:
    """
    # print(img_path, label_path)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.reshape(tensor=label, shape=(512, 512))
    label = tf.cast(label, tf.float32) / 255.0
    # label = tf.cast(label, tf.uint8)
    # label = tf.one_hot(indices=label, depth=2, on_value=1, off_value=0)

    print(image.shape, label.shape)
    return image, label

