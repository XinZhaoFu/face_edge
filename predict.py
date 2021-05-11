# coding=utf-8
import glob
import cv2
from model.unet import UNet
from model.bisenetv2 import BisenetV2
import tensorflow as tf
import numpy as np
import datetime

np.set_printoptions(threshold=np.inf)


def predict(checkpoint_save_path, test_file_path, predict_save_path):
    print('[info]模型加载 图片加载')
    # 加载模型
    # model = BisenetV2(detail_filters=32, aggregation_filters=32, final_filters=2)
    model = UNet(filters=32, num_class=2, semantic_num_cbr=1, detail_num_cbr=4)
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.load_weights(checkpoint_save_path)

    test_file_path_list = glob.glob(test_file_path+'*.jpg')
    print(len(test_file_path_list))
    test_img_list = np.empty(shape=(len(test_file_path_list), 512, 512, 3), dtype=np.float32)
    test_img_name_list = []
    for index, test_file in enumerate(test_file_path_list):
        test_img = cv2.imread(test_file)
        test_img = cv2.resize(test_img, dsize=(512, 512))
        test_img = test_img / 255.
        test_img_list[index, :, :, :] = test_img[:, :, :]
        test_img_name = (test_file.split('/')[-1]).split('.')[0]
        test_img_name = "{:0>5d}".format(int(test_img_name))
        test_img_name_list.append(test_img_name)

    print('[info]执行推理')
    predict_list = model.predict(test_img_list)
    # print(predict_temp.shape)

    print('[info]图片存储')
    for index in range(len(test_file_path_list)):
        predict_img = np.empty(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict_list[index, :, :, 1] * 255
        # predict_img = cv2.blur(predict_img, ksize=(5, 5))
        # predict_img = cv2.adaptiveThreshold(predict_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
        _, predict_img = cv2.threshold(predict_img, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(predict_save_path+test_img_name_list[index]+'_nose.png', predict_img)


if __name__ == '__main__':
    # checkpoint_save_path = './checkpoint/face_edge.ckpt'
    checkpoint_save_path = './checkpoint/detail_con_unet_nose_edge.ckpt'
    # checkpoint_save_path = './checkpoint/bisev2_3x2_face_edge.ckpt.index'

    test_file_path = './data/celeb_ori_img/'
    predict_save_path = './data/celeb_ori_label/'

    start_time = datetime.datetime.now()

    predict(checkpoint_save_path, test_file_path, predict_save_path)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])
