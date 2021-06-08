import cv2
import numpy as np
from random import randint, uniform


def random_crop(img, label):
    img_rows, img_cols, img_channel = img.shape
    label_rows, label_cols = label.shape

    if img_rows == label_rows and img_cols == label_cols:
        if img_rows >= img_cols:
            random_crop_length = randint(img_rows // 4, img_rows // 2)
        else:
            random_crop_length = randint(img_cols // 4, img_cols // 2)

        random_crop_row_init = randint(1, img_rows - random_crop_length)
        random_crop_col_init = randint(1, img_cols - random_crop_length)

        crop_img = np.empty(shape=(random_crop_length, random_crop_length, img_channel), dtype=np.uint8)
        crop_label = np.empty(shape=(random_crop_length, random_crop_length), dtype=np.uint8)

        crop_img[:, :, :] = img[random_crop_row_init:random_crop_row_init + random_crop_length,
                                random_crop_col_init:random_crop_col_init + random_crop_length, :]
        crop_label[:, :] = label[random_crop_row_init:random_crop_row_init + random_crop_length,
                                 random_crop_col_init:random_crop_col_init + random_crop_length]
    else:
        resize_rate = label_rows / img_rows
        assert resize_rate == label_cols / img_cols

        if img_rows >= img_cols:
            random_crop_img_length = randint(img_rows // 4, img_rows // 2)
        else:
            random_crop_img_length = randint(img_cols // 4, img_cols // 2)
        random_crop_label_length = random_crop_img_length * resize_rate

        random_crop_img_row_init = randint(1, img_rows - random_crop_img_length)
        random_crop_img_col_init = randint(1, img_cols - random_crop_img_length)

        random_crop_label_row_init = randint(1, img_rows - random_crop_label_length)
        random_crop_label_col_init = randint(1, img_cols - random_crop_label_length)

        crop_img = np.empty(shape=(random_crop_img_length, random_crop_img_length, img_channel), dtype=np.uint8)
        crop_label = np.empty(shape=(random_crop_label_length, random_crop_label_length), dtype=np.uint8)

        crop_img[:, :, :] = img[random_crop_img_row_init:random_crop_img_row_init + random_crop_img_length,
                                random_crop_img_col_init:random_crop_img_col_init + random_crop_img_length, :]
        crop_label[:, :] = label[random_crop_label_row_init:random_crop_label_row_init + random_crop_label_length,
                                 random_crop_label_col_init:random_crop_label_col_init + random_crop_label_length]

    return crop_img, crop_label


def random_color_scale(img, alpha_rate=0.2, base_beta=15):
    alpha = uniform(1-alpha_rate, 1+alpha_rate)
    beta = randint(0-base_beta, base_beta)
    img = img * alpha + beta

    return img


def flip(img, label):
    img = cv2.flip(img, 1)
    label = cv2.flip(label, 1)

    return img, label


def gridMask(img, rate=0.1):
    """
    对图片进行gridmask 每行每列各十个 以边均匀十等分 每一长度中包含mask长度、offset偏差和留白
    长方形需要改一下
    盲猜，过拟合增大，欠拟合缩小，自行调节

    :param img: 输入应为正方形图像
    :param rate: mask长度与十分之一边长的比值
    :return: gridmask后的图像
    """
    img_length, _, channel = img.shape
    fill_img_length = int(img_length + 0.2 * img_length)
    offset = randint(0, int(0.1 * fill_img_length))
    mask_length = int(0.1 * fill_img_length * rate)
    fill_img = np.zeros((fill_img_length, fill_img_length, channel))
    fill_img[int(0.1 * img_length):int(0.1 * img_length) + img_length,
             int(0.1 * img_length):int(0.1 * img_length) + img_length] = img
    for width_num in range(10):
        for length_num in range(10):
            length_base_patch = int(0.1 * fill_img_length * length_num) + offset
            width_base_patch = int(0.1 * fill_img_length * width_num) + offset
            fill_img[length_base_patch:length_base_patch + mask_length,
                     width_base_patch:width_base_patch + mask_length, :] = 0
    img = fill_img[int(0.1 * img_length):int(0.1 * img_length) + img_length,
                   int(0.1 * img_length):int(0.1 * img_length) + img_length, :]

    return img
