# coding=utf-8
import face_recognition
import numpy as np
import os
import shutil
import cv2
from glob import glob
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
                           is_recreate_dir=False,
                           resize=512,
                           is_del_aug=False):
    """
    将img和label从一文件夹转至其他位置

    :param is_del_aug:
    :param resize:
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

    for img_file_path, label_file_path in tqdm(zip(distribution_img_file_list, distribution_label_file_list),
                                               total=len(distribution_img_file_list)):
        img_name = img_file_path.split('/')[-1]
        label_name = label_file_path.split('/')[-1]
        if is_del_aug and '_' in img_name:
            continue

        if resize == 0:
            shutil.copyfile(img_file_path, distribution_img_file_path + img_name)
            shutil.copyfile(label_file_path, distribution_label_file_path + label_name)
        else:
            img = cv2.imread(img_file_path)
            img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            label = cv2.imread(label_file_path)
            label = cv2.resize(label, dsize=(resize, resize), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(distribution_img_file_path + img_name, img)
            cv2.imwrite(distribution_label_file_path + label_name, label)


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
        print('[INFO] 新建文件夹：' + folder_name)


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
        os.makedirs(folder_name)
    print('[INFO] 重建文件夹：' + folder_name)


def check_img_label_list(img_file_path_list, label_file_path_list):
    """
    校验文件是否对应

    :param img_file_path_list:
    :param label_file_path_list:
    :return:
    """
    for img_path, label_path in zip(img_file_path_list, label_file_path_list):
        img_name = (img_path.split('/')[-1]).split('.')[0]
        label_name = (label_path.split('/')[-1]).split('.')[0]

        assert img_name == label_name
    print('[INFO] 文件对应检查通过')


def color_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    planes = cv2.split(img)
    for i in range(0, 3):
        planes[i] = clahe.apply(planes[i])
    img = cv2.merge(planes)

    return img


def clean_val_file(val_file_path, val_type='*.png'):
    val_file_list = glob(val_file_path + val_type)
    for val_file in tqdm(val_file_list):
        val_file_name = (val_file.split('/')[-1]).split('.')[0]
        if '_' in val_file_name:
            os.remove(val_file)


def check_file_is_aug(img_file_path_list, label_file_path_list):
    """
    对传入的文件是否含有数据增强的文件做一下判断 并传入分别的list中

    :param img_file_path_list:
    :param label_file_path_list:
    :return:
    """
    print('[INFO] 数据分组')
    normal_img_file_path_list, normal_label_file_path_list = [], []
    aug_img_file_path_list, aug_label_file_path_list = [], []
    for img_file_path, label_file_path in tqdm(zip(img_file_path_list, label_file_path_list),
                                               total=len(img_file_path_list)):
        img_file_name = (img_file_path.split('/')[-1]).split('.')[0]
        label_file_name = (label_file_path.split('/')[-1]).split('.')[0]
        if img_file_name != label_file_name:
            break
        if '_' in img_file_name:
            aug_img_file_path_list.append(img_file_path)
            aug_label_file_path_list.append(label_file_path)
        else:
            normal_img_file_path_list.append(img_file_path)
            normal_label_file_path_list.append(label_file_path)

    return normal_img_file_path_list, normal_label_file_path_list, aug_img_file_path_list, aug_label_file_path_list


def face_crop(img):
    img_rows, img_cols, img_channel = img.shape
    if img_rows < 512 or img_cols < 512:
        img = cv2.resize(img, (512, 512))

    filling_img = filling(img)
    location = face_recognition.face_locations(filling_img)
    if len(location) == 0:
        return img

    [(x1, y2, x2, y1)] = location
    # crop_x1 = x1 - (x2 - x1) // 2
    # crop_x2 = x2 + (x2 - x1) // 2
    # crop_y1 = y1 - (y2 - y1) // 2
    # crop_y2 = y2 + (y2 - y1) // 2

    crop_x1 = max(x1 - (x2 - x1) // 2, 256)
    crop_x2 = min(x2 + (x2 - x1) // 2, img_rows + 256)
    crop_y1 = max(y1 - (y2 - y1) // 2, 256)
    crop_y2 = min(y2 + (y2 - y1) // 2, img_cols + 256)
    if crop_y2 - crop_y1 > crop_x2 - crop_x1:
        crop_y2 = (crop_y2 + crop_y1) // 2 + (crop_x2 - crop_x1) // 2
        crop_y1 = (crop_y2 + crop_y1) // 2 - (crop_x2 - crop_x1) // 2
    if crop_x2 - crop_x1 > crop_y2 - crop_y1:
        crop_x2 = (crop_x2 + crop_x1) // 2 + (crop_y2 - crop_y1) // 2
        crop_x1 = (crop_x2 + crop_x1) // 2 - (crop_y2 - crop_y1) // 2

    img = filling_img[crop_x1:crop_x2, crop_y1:crop_y2, :]

    return img


def filling(img):
    """
    :param img:
    :return:
    """
    img_rows, img_cols, img_channel = img.shape
    filling_img = np.zeros(shape=(img_rows + 512, img_cols + 512, 3), dtype=np.uint8)
    filling_img[256:img_rows+256, 256:img_cols+256] = img[:, :]

    return filling_img
