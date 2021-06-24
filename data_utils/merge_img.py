import cv2
from glob import glob
import numpy as np

img_list = glob('../data/res/con_temp/' + '*.jpg')
label_list = glob('../data/res/con_temp/' + '*.png')
img_list.sort()
label_list.sort()

label_list = label_list[:15]

for img_file, label_file in zip(img_list, label_list):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (512, 512))
    img_name = img_file.split('/')[-1]
    label = cv2.imread(label_file, 0)

    (rows, cols) = np.where(label > 64)
    img[rows, cols, :] = 255

    cv2.imwrite('../data/res/merge/' + img_name, img)
