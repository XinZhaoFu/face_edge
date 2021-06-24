import cv2
import face_recognition
from glob import glob
from tqdm import tqdm
import numpy as np


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


def main():
    img_path = '../data/res/sample/'
    save_path = '../data/res/crop/'

    img_file_list = glob(img_path + '*.jpg')
    for img_file_path in tqdm(img_file_list):
        img = cv2.imread(img_file_path)
        img_name = img_file_path.split('/')[-1]
        img = face_crop(img)
        cv2.imwrite(save_path + img_name, img)


if __name__ == '__main__':
    main()
