# coding=utf-8
import glob
import cv2
from model.unet import UNet
from model.bisenetv2 import BisenetV2
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from loss.loss import dice_loss, binary_focal_loss


np.set_printoptions(threshold=np.inf)


gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def predict(checkpoint_save_path, test_file_path, predict_save_path):
    """
    这里batchsize形同为1  需要一批多个的话得改一下

    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型
    # model = BisenetV2(detail_filters=32, aggregation_filters=32, final_filters=2)
    model = UNet(filters=32, num_class=1, semantic_num_cbr=1, detail_num_cbr=4, end_activation='sigmoid')
    model.compile(optimizer='Adam',
                  loss=binary_focal_loss(),
                  metrics=['accuracy'])
    model.load_weights(checkpoint_save_path)

    test_file_path_list = glob.glob(test_file_path+'*.jpg')
    print(len(test_file_path_list))
    test_img_np = np.empty(shape=(1, 512, 512, 3), dtype=np.float32)

    test_file_path_list = tqdm(test_file_path_list)
    for test_file in test_file_path_list:
        test_img = cv2.imread(test_file)
        test_img = cv2.resize(test_img, dsize=(512, 512))
        test_img = test_img / 255.
        test_img_np[0, :, :, :] = test_img[:, :, :]
        test_img_name = (test_file.split('/')[-1]).split('.')[0]
        # test_img_name = "{:0>5d}".format(int(test_img_name)) + '_nose.png'
        test_img_name = test_img_name + '_dice_test.png'

        predict_temp = model.predict(test_img_np)

        predict_img = np.empty(shape=(512, 512), dtype=np.uint8)

        predict_img[:, :] = predict_temp[0, :, :, 0] * 255

        # _, predict_img = cv2.threshold(predict_img, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(predict_save_path + test_img_name, predict_img)
        test_file_path_list.set_description('生成中')


def main():
    # checkpoint_save_path = './checkpoint/face_edge.ckpt'
    checkpoint_save_path = './checkpoint/detail_con_unet_face_edge.ckpt'
    # checkpoint_save_path = './checkpoint/bisev2_3x2_face_edge.ckpt.index'

    test_file_path = './data/test/'
    predict_save_path = './data/predict/'

    start_time = datetime.datetime.now()

    predict(checkpoint_save_path, test_file_path, predict_save_path)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
