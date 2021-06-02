# coding=utf-8
import glob
import cv2
from model.dense_unet import DenseUNet
from model.densenet import DenseNet
from model.bisenetv2 import BisenetV2
from model.unet import UNet
from model.u2net import U2Net
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from loss.loss import dice_loss, binary_focal_loss, mix_dice_focal_loss


np.set_printoptions(threshold=np.inf)


gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_comple):
    """
    这里batchsize形同为1  需要一批多个的话得改一下

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
    # model = BisenetV2(detail_filters=32, aggregation_filters=32, num_class=1, final_act='sigmoid')
    # model = DenseUNet(semantic_filters=16,
    #                   detail_filters=32,
    #                   num_class=1,
    #                   semantic_num_cbr=1,
    #                   detail_num_cbr=2,
    #                   end_activation='sigmoid')
    # model = DenseNet(filters=32, num_class=1, activation='sigmoid')
    # model = UNet(semantic_filters=16,
    #              detail_filters=32,
    #              num_class=1,
    #              semantic_num_cbr=1,
    #              detail_num_cbr=6,
    #              end_activation='sigmoid')
    model.compile(optimizer='Adam',
                  loss=dice_loss(),
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
        test_img_name = img_name_comple + '_' + ex_info + '_' + test_img_name + '.png'

        predict_temp = model.predict(test_img_np)

        predict_img = np.empty(shape=(512, 512), dtype=np.uint8)

        predict_img[:, :] = predict_temp[0, :, :, 0] * 255

        # _, predict_img = cv2.threshold(predict_img, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(predict_save_path + test_img_name, predict_img)
        test_file_path_list.set_description('生成中')


def main():
    # ex_info = 'dense_unet_df32sf16_mix_loss'
    # ex_info = 'detail_con_unet_face_edge_focal'
    # ex_info = 'bisev2_mix_loss'
    ex_info = 'u2net_mix_loss'

    checkpoint_save_path = './checkpoint/' + ex_info + '.ckpt'

    test_file_path = './data/test/'
    predict_save_path = './data/predict/'

    start_time = datetime.datetime.now()

    tran_tab = str.maketrans('- :.', '____')
    img_name_comple = str(start_time).translate(tran_tab)

    predict(checkpoint_save_path, test_file_path, predict_save_path, ex_info, img_name_comple)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
