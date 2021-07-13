import cv2
from glob import glob
import numpy as np


def merge_img_label():
    img_list = glob('../data/res/con_temp/' + '*.jpg')
    label_list = glob('../data/res/con_temp/' + '*.png')
    img_list.sort()
    label_list.sort()

    for img_file, label_file in zip(img_list, label_list):
        img = cv2.imread(img_file)
        img = cv2.resize(img, (512, 512))
        img_name = img_file.split('/')[-1]
        label = cv2.imread(label_file, 0)

        (rows, cols) = np.where(label > 64)
        img[rows, cols - 50, :] = 255

        cv2.imwrite('../data/res/merge/' + img_name, img)


def merge_img():
    img_list = glob('../data/res/merge/' + '*.jpg')
    img_list.sort()
    label_list = glob('../data/res/merge/' + '*.png')
    label_list.sort()

    res_img = np.empty(shape=(512 * 7, 512 * 3, 3), dtype=np.uint8)

    for index in range(7):
        img1 = cv2.imread(img_list[index])
        img2 = cv2.imread(img_list[index + 7])
        label = cv2.imread(label_list[index])

        img1 = cv2.resize(img1, dsize=(512, 512))
        img2 = cv2.resize(img2, dsize=(512, 512))
        label = cv2.resize(label, dsize=(512, 512))

        res_img[512 * index:512 * (index + 1), 0:512] = img1
        res_img[512 * index:512 * (index + 1), 512:1024] = img2
        res_img[512 * index:512 * (index + 1), 1024:1536] = label

    cv2.imwrite('../data/res/merge/res_img.jpg', res_img)


if __name__ == '__main__':
    merge_img()
