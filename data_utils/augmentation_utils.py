from random import randint, uniform

import cv2
import numpy as np


def random_crop(img, label):
    """
    随机裁剪  因为img尺寸和label尺寸不一定对应 resize_rate是用于调整比例的

    :param img:
    :param label:
    :return:
    """
    img_rows, img_cols, img_channel = img.shape
    label_rows, label_cols = label.shape

    resize_rate_row = label_rows / img_rows
    resize_rate_col = label_cols / img_cols

    img_random_crop_row_length = randint(img_rows // 4, (img_rows // 4) * 3)
    img_random_crop_col_length = randint(img_cols // 4, (img_cols // 4) * 3)
    label_random_crop_row_length = int(img_random_crop_row_length * resize_rate_row)
    label_random_crop_col_length = int(img_random_crop_col_length * resize_rate_col)

    img_random_crop_row_init = randint(1, img_rows - img_random_crop_row_length)
    img_random_crop_col_init = randint(1, img_cols - img_random_crop_col_length)
    label_random_crop_row_init = int(img_random_crop_row_init * resize_rate_row)
    label_random_crop_col_init = int(img_random_crop_col_init * resize_rate_col)

    crop_img = np.empty(shape=(img_random_crop_row_length, img_random_crop_col_length, img_channel), dtype=np.uint8)
    crop_label = np.empty(shape=(label_random_crop_row_length, label_random_crop_col_length), dtype=np.uint8)

    crop_img[:, :, :] = img[img_random_crop_row_init:img_random_crop_row_init + img_random_crop_row_length,
                            img_random_crop_col_init:img_random_crop_col_init + img_random_crop_col_length, :]
    crop_label[:, :] = label[label_random_crop_row_init: label_random_crop_row_init + label_random_crop_row_length,
                             label_random_crop_col_init: label_random_crop_col_init + label_random_crop_col_length]

    return crop_img, crop_label


def random_color_scale(img, alpha_rate=0.2, base_beta=15):
    """
    做一个颜色扰动 img = img * alpha + beta

    :param img:
    :param alpha_rate:
    :param base_beta:
    :return:
    """
    alpha = uniform(1 - alpha_rate, 1 + alpha_rate)
    beta = randint(0 - base_beta, base_beta)
    img = img * alpha + beta

    return img


def random_flip(img, label):
    """
    做一个随机翻转

    :param img:
    :param label:
    :return:
    """
    random_num = randint(0, 1)
    img = cv2.flip(img, random_num)
    label = cv2.flip(label, random_num)

    return img, label


def random_rotate(img, label):
    """
    做一个随机旋转 但只有90 180 和270

    :param img:
    :param label:
    :return:
    """
    img_rows, img_cols, _ = img.shape
    label_rows, label_cols = label.shape
    random_num = randint(1, 3)
    img_rotate = cv2.getRotationMatrix2D((img_rows * 0.5, img_cols * 0.5), 90 * random_num, 1)
    label_rotate = cv2.getRotationMatrix2D((label_rows * 0.5, label_cols * 0.5), 90 * random_num, 1)
    if random_num == 2:
        img = cv2.warpAffine(img, img_rotate, (img_rows, img_cols))
        label = cv2.warpAffine(label, label_rotate, (label_rows, label_cols))
    else:
        img = cv2.warpAffine(img, img_rotate, (img_cols, img_rows))
        label = cv2.warpAffine(label, label_rotate, (label_cols, label_rows))

    return img, label


def cutout(img, mask_rate=0.3):
    """
    对图片进行cutout 遮盖位置随机
    遮盖长度为空时用默认值
    过拟合增大，欠拟合缩小，自行调节
    添加遮盖前 对图像一圈进行0填充

    :param img:
    :param mask_rate:
    :return:cutout后的图像
    """
    img_rows, img_cols, img_channel = img.shape
    mask_rows = int(img_rows * mask_rate)
    mask_cols = int(img_cols * mask_rate)
    region_x, region_y = randint(0, img_rows + mask_rows), randint(0, img_cols + mask_cols)

    fill_img = np.zeros((img_rows + mask_rows * 2, img_cols + mask_cols * 2, img_channel))
    fill_img[mask_rows:mask_rows + img_rows, mask_cols:mask_cols + img_cols] = img
    fill_img[region_x:region_x + mask_rows, region_y:region_y + mask_cols] = 0
    img = fill_img[mask_rows:mask_rows + img_rows, mask_cols:mask_cols + img_cols]

    return img


def gridMask(img, rate=0.1):
    """
    对图片进行gridmask
    每行每列各十个 以边均匀十等分 共计10*10个单元块  在每一个单元块内进行遮盖
    每一单元块其边长 = mask长度、offset偏差和留白
    长方形没做适配
    若干次的经验来看，过拟合增大rate，欠拟合缩小rate，但毕竟只是经验值

    :param img:
    :param rate: mask长度与一单元块边长的比值
    :return: gridmask后的图像
    """
    img_rows, img_cols, img_channel = img.shape
    fill_img_rows_length = int(img_rows + 0.2 * img_rows)
    fill_img_cols_length = int(img_cols + 0.2 * img_cols)
    rows_offset = randint(0, int(0.1 * fill_img_rows_length))
    cols_offset = randint(0, int(0.1 * fill_img_cols_length))
    rows_mask_length = int(0.1 * fill_img_rows_length * rate)
    cols_mask_length = int(0.1 * fill_img_cols_length * rate)

    fill_img = np.zeros((fill_img_rows_length, fill_img_cols_length, img_channel))
    fill_img[int(0.1 * img_rows):int(0.1 * img_rows) + img_rows,
             int(0.1 * img_cols):int(0.1 * img_cols) + img_cols] = img

    for width_num in range(10):
        for length_num in range(10):
            length_base_patch = int(0.1 * fill_img_rows_length * length_num) + rows_offset
            width_base_patch = int(0.1 * fill_img_cols_length * width_num) + cols_offset
            fill_img[length_base_patch:length_base_patch + rows_mask_length,
                     width_base_patch:width_base_patch + cols_mask_length] = 0

    img = fill_img[int(0.1 * img_rows):int(0.1 * img_rows) + img_rows,
                   int(0.1 * img_cols):int(0.1 * img_cols) + img_cols]

    return img


def random_filling(img, label):
    """
    目标是在图片周围填充一圈空值  实现方式是生成一个大的空值图 找一随机位置将原图填进去

    :param img:
    :param label:
    :return:
    """
    img = cv2.resize(img, dsize=(256, 256))
    label = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    # _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)

    filling_img = np.zeros(shape=(512, 512, 3))
    filling_label = np.zeros(shape=(512, 512))

    random_x, random_y = randint(0, 256), randint(0, 256)

    filling_img[random_x:random_x + 256, random_y:random_y + 256] = img
    filling_label[random_x:random_x + 256, random_y:random_y + 256] = label

    return filling_img, filling_label


def get_augmentation(img,
                     con_label,
                     save_img_path,
                     save_label_path,
                     img_name,
                     label_name,
                     random_crop_num=1,
                     gridmask_num=1,
                     cutout_num=1,
                     random_filling_num=1):
    """
    数据扩增

    :param random_filling_num:
    :param cutout_num:
    :param gridmask_num:
    :param random_crop_num:
    :param img:
    :param con_label:
    :param save_img_path:
    :param save_label_path:
    :param img_name:
    :param label_name:
    :return:
    """
    # random_crop
    for index in range(random_crop_num):
        crop_img, crop_label = random_crop(img, con_label)
        cv2.imwrite(save_img_path + img_name + '_random_crop_' + str(index) + '.jpg', crop_img)
        cv2.imwrite(save_label_path + label_name + '_random_crop_' + str(index) + '.png', crop_label)

    # gridmask
    for index in range(gridmask_num):
        gridmask_img = gridMask(img, rate=0.1)
        gridmask_img, gridmask_label = random_rotate(gridmask_img, con_label)
        gridmask_img, gridmask_label = random_flip(gridmask_img, gridmask_label)
        cv2.imwrite(save_img_path + img_name + '_gridmask_' + str(index) + '.jpg', gridmask_img)
        cv2.imwrite(save_label_path + label_name + '_gridmask_' + str(index) + '.png', gridmask_label)

    # cutout
    for index in range(cutout_num):
        cutout_img = cutout(img, mask_rate=0.3)
        cutout_img, cutout_label = random_rotate(cutout_img, con_label)
        cutout_img, cutout_label = random_flip(cutout_img, cutout_label)
        cv2.imwrite(save_img_path + img_name + '_cutout_' + str(index) + '.jpg', cutout_img)
        cv2.imwrite(save_label_path + label_name + '_cutout_' + str(index) + '.png', cutout_label)

    # random_filling
    for index in range(random_filling_num):
        filling_img, filling_label = random_filling(img, con_label)
        cv2.imwrite(save_img_path + img_name + '_random_filling_' + str(index) + '.jpg', filling_img)
        cv2.imwrite(save_label_path + label_name + '_random_filling_' + str(index) + '.png', filling_label)