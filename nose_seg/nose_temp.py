from glob import glob
import cv2
import numpy as np
import tqdm

save_path = '../data/lapa_nose_label/'
label_path = '../data/lapa_ori_label/'

label_file_path_list = glob(label_path+'*.png')
print(len(label_file_path_list))

label_file_path_list = tqdm.tqdm(label_file_path_list)
for label_file_path in label_file_path_list:
    label_name = label_file_path.split('/')[-1]

    label = cv2.imread(label_file_path, 0)
    rows, cols = label.shape
    nose_label = np.zeros(shape=label.shape, dtype=np.uint8)
    label = np.array(label, dtype=np.uint8)

    for p_value in np.nditer(label, op_flags=['readwrite']):
        if p_value == 6:
            p_value[...] = 255
        else:
            p_value[...] = 0

    cv2.imwrite(save_path+label_name, label)
    label_file_path_list.set_description("Processing %s" % label_name)
