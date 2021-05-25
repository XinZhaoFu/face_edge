# coding=utf-8
import numpy as np
import os
import shutil
from tqdm import tqdm


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


def distribution_img_label(distribution_img_file_list,
                           distribution_label_file_list,
                           distribution_img_file_path,
                           distribution_label_file_path,
                           is_recreate_dir=False):
    """
    将img和label从一文件夹转至其他位置

    :param is_recreate_dir:
    :param distribution_img_file_list:
    :param distribution_label_file_list:
    :param distribution_img_file_path:
    :param distribution_label_file_path:
    :return:
    """
    if is_recreate_dir:
        recreate_dir(distribution_img_file_path)
        recreate_dir(distribution_label_file_path)

    assert len(distribution_img_file_list) == len(distribution_label_file_list)

    for index in tqdm(range(len(distribution_img_file_list))):
        img_file = distribution_img_file_list[index]
        label_file = distribution_label_file_list[index]
        img_name = img_file.split('/')[-1]
        label_name = label_file.split('/')[-1]

        shutil.copyfile(img_file, distribution_img_file_path + img_name)
        shutil.copyfile(label_file, distribution_label_file_path + label_name)


def distribution_file(file_path_list, target_file_path, is_recreate_dir=False):
    """
    将文件路径列表中的文件复制到目标文件夹

    :param is_recreate_dir:
    :param file_path_list:
    :param target_file_path:
    :return:
    """
    if is_recreate_dir:
        recreate_dir(target_file_path)

    for index in tqdm(range(len(file_path_list))):
        file_path = file_path_list[index]
        file_name = file_path.split('/')[-1]
        shutil.copyfile(file_path, target_file_path + file_name)


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
    """
    校验文件是否对应

    :param img_file_list:
    :param label_file_list:
    :return:
    """
    for img_path, label_path in zip(img_file_list, label_file_list):
        img_name = (img_path.split('/')[-1]).split('.')[0]
        label_name = (label_path.split('/')[-1]).split('.')[0]

        assert img_name == label_name
    print('文件对应检查通过')
