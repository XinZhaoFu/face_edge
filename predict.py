# coding=utf-8
import glob
import cv2
import os
from model.dense_unet import DenseUNet
from model.densenet import DenseNet
from model.bisenetv2 import BisenetV2
from model.unet import UNet
from model.u2net import U2Net
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from data_utils.utils import face_crop, edge_smooth

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(threshold=np.inf)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def edge_predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_complement):
    """
    这里batchsize形同为1  需要一批多个的话得改一下

    :param img_name_complement:
    :param ex_info:
    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型
    model = U2Net(rsu_middle_filters=16,
                  rsu_out_filters=32,
                  num_class=1,
                  end_activation='sigmoid',
                  only_output=True)

    model.load_weights(checkpoint_save_path)

    test_file_path_list = glob.glob(test_file_path + '*.jpg')

    for test_file in tqdm(test_file_path_list):
        test_img, test_img_name = get_predict_img(test_file)
        test_img_name = img_name_complement + '_' + ex_info + '_' + test_img_name + '.png'

        predict_temp = model.predict(test_img)

        predict_img = np.empty(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict_temp[0, :, :, 0] * 255

        # predict_rgb_img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
        # predict_rgb_img[:, :, 2] = predict_img

        (rows, cols) = np.where(predict_img > 128)
        predict_img[rows, cols] = 255

        predict_img = edge_smooth(predict_img)

        cv2.imwrite(predict_save_path + test_img_name, predict_img)
        # cv2.imwrite(predict_save_path + test_img_name, predict_rgb_img)


def seg_predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_complement):
    """
    这里batchsize形同为1  需要一批多个的话得改一下

    :param img_name_complement:
    :param ex_info:
    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型

    model = U2Net(rsu_middle_filters=16,
                  rsu_out_filters=32,
                  num_class=20,
                  end_activation='softmax',
                  only_output=True)

    model.load_weights(checkpoint_save_path)

    test_file_path_list = glob.glob(test_file_path + '*.jpg')

    for test_file in tqdm(test_file_path_list):
        test_img, test_img_name = get_predict_img(test_file)
        test_img_name = img_name_complement + '_' + ex_info + '_' + test_img_name + '.png'

        predict_temp = model.predict(test_img)

        predict_temp = tf.math.argmax(predict_temp, 3)
        predict_temp = np.array(predict_temp)
        predict_temp = np.reshape(predict_temp, newshape=(512, 512))
        predict_img = np.empty(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict_temp[:, :] * 12
        # predict_img = cv2.Canny(predict_img, 0, 0)

        cv2.imwrite(predict_save_path + test_img_name, predict_img)


def get_predict_img(test_file_path):
    """
    对人脸进行裁剪 并将数据格式与网络输入相对应

    :param test_file_path:
    :return:
    """
    test_img_name = (test_file_path.split('/')[-1]).split('.')[0]
    test_img = cv2.imread(test_file_path)
    test_img = face_crop(test_img)

    test_img = cv2.resize(test_img, dsize=(512, 512))
    test_img = np.array(test_img / 255.)
    test_img = np.reshape(test_img, newshape=(1, 512, 512, 3))

    return test_img, test_img_name


def main():
    # ex_info = 'dense_unet_df32sf16_mix_loss'
    # ex_info = 'detail_con_unet_face_edge_focal'
    # ex_info = 'bisev2_mix_loss'
    # ex_info = 'u2net_mix_loss'
    # ex_info = 'u2net_16_64'
    # ex_info = 'u2net_16_64_bin'
    # ex_info = 'u2net_dice'
    # ex_info = 'u2net_dice_02aug30000'
    # ex_info = 'u2net_bin_02aug10000'
    # ex_info = 'u2net_dice_02aug42000'
    ex_info = 'u2net_seg'

    checkpoint_save_path = './checkpoint/' + ex_info + '.ckpt'

    test_file_path = './data/res/sample/'
    predict_save_path = './data/res/predict3/'

    start_time = datetime.datetime.now()

    tran_tab = str.maketrans('- :.', '____')
    img_name_complement = str(start_time).translate(tran_tab)

    # edge_predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_complement)
    seg_predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_complement)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
