import cv2
from glob import glob
from tqdm import tqdm

from data_utils.utils import face_crop


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
