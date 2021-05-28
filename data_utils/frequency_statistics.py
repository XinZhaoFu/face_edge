from glob import glob
import cv2
import numpy as np
from tqdm import tqdm

label_path = '../data/celeb_edge/'
label_file_list = glob(label_path + '*.png')
points_255_sum = 0
points_0_sum = 0

label_file_list = tqdm(label_file_list)
for label_file in label_file_list:
    label = cv2.imread(label_file, 0)

    points_255 = np.where(label == 255)
    points_255_sum += len(points_255[0])

    points_0 = np.where(label == 0)
    points_0_sum += len(points_0[0])

print((1 / points_0_sum) * ((points_0_sum + points_255_sum) / 2.0),
      (1 / points_255_sum) * ((points_0_sum + points_255_sum) / 2.0))
"""
[0.5073703301521681, 34.419783081421244]
"""
