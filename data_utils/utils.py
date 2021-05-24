# coding=utf-8
import numpy as np
import os
import shutil


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


def distribution_img_label(dis_img_file_list,
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


def distribution_file(file_path_list, target_file_path):
    """

    :param file_path_list:
    :param target_file_path:
    :return:
    """
    for file_path in file_path_list:
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
    for img_path, label_path in zip(img_file_list, label_file_list):
        img_name = (img_path.split('/')[-1]).split('.')[0]
        label_name = (label_path.split('/')[-1]).split('.')[0]

        assert img_name == label_name
    print('文件对应检查通过')
