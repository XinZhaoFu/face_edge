# coding=utf-8
import os
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision
from model.dense_unet import DenseUNet
from model.densenet import DenseNet
from model.unet import UNet
from model.bisenetv2 import BisenetV2
from model.u2net import U2Net
from data_utils.dataloader import Data_Loader_File
import matplotlib.pyplot as plt
import pandas as pd
import setproctitle
import numpy as np
from loss.loss import binary_focal_loss, dice_loss, mix_dice_focal_loss, binary_crossentropy_weight, \
    u2net_bce_loss, dice_bce_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

setproctitle.setproctitle("face_edge")

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('[INFO] 计算类型: %s' % policy.compute_dtype)
print('[INFO] 变量类型: %s' % policy.variable_dtype)


def parseArgs():
    """
    获得参数

    :return:
    """
    parser = argparse.ArgumentParser(description='face edge demo')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='learning_rate',
                        default=0,
                        type=float)
    parser.add_argument('--epochs',
                        dest='epochs',
                        help='epochs',
                        default=1,
                        type=int)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='batch_size',
                        default=4,
                        type=int)
    parser.add_argument('--load_train_file_number',
                        dest='load_train_file_number',
                        help='load_train_file_number',
                        default=0,
                        type=int)
    parser.add_argument('--load_val_file_number',
                        dest='load_val_file_number',
                        help='load_val_file_number',
                        default=0,
                        type=int)
    parser.add_argument('--load_weights',
                        dest='load_weights',
                        help='load_weights type is boolean',
                        default=False, type=bool)
    parser.add_argument('--data_augmentation',
                        dest='data_augmentation',
                        help='data_augmentation type is float, range is 0 ~ 1',
                        default=0,
                        type=float)
    args = parser.parse_args()
    return args


class train:
    def __init__(self,
                 load_weights=False,
                 batch_size=128,
                 epochs=0,
                 load_train_file_number=0,
                 load_val_file_number=0,
                 data_augmentation=0,
                 augmentation_rate=1,
                 erase_rate=0.1,
                 learning_rate=0,
                 train_file_path='./data/train/',
                 val_file_path='./data/val/',
                 ex_info='info',
                 checkpoint_save_path='./checkpoint/'):
        self.load_weights = load_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.load_train_file_number = load_train_file_number
        self.load_val_file_number = load_val_file_number
        self.erase_rate = erase_rate
        self.augmentation_rate = augmentation_rate
        self.learning_rate = learning_rate
        self.checkpoint_save_path = checkpoint_save_path + ex_info + '.ckpt'
        # self.checkpoint_input_path = './checkpoint/' + 'u2net_dice_02aug42000' + '.ckpt'
        self.checkpoint_input_path = self.checkpoint_save_path
        print('[INFO] checkpoint_input_path:\t' + self.checkpoint_input_path)

        # self.strategy = tf.distribute.MirroredStrategy()
        # print('[INFO] 目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))

        data_loader = Data_Loader_File(data_augmentation=self.data_augmentation,
                                       batch_size=self.batch_size,
                                       train_file_path=train_file_path,
                                       val_file_path=val_file_path)
        self.train_datasets = data_loader.load_train_data(load_file_number=self.load_train_file_number)
        self.val_datasets = data_loader.load_val_data(load_file_number=self.load_val_file_number)
        self.class_weight = np.array([0.5073703301521681, 34.419783081421244])

    def model_train(self):
        """
        可多卡训练

        :return:
        """
        # with self.strategy.scope():
        # model = DenseUNet(semantic_filters=16,
        #                   detail_filters=32,
        #                   num_class=1,
        #                   semantic_num_cbr=1,
        #                   detail_num_cbr=3,
        #                   end_activation='sigmoid')

        # model = DenseNet(filters=64, num_class=1, activation='sigmoid')

        # model = UNet(semantic_filters=16,
        #              detail_filters=32,
        #              num_class=1,
        #              semantic_num_cbr=1,
        #              detail_num_cbr=4,
        #              end_activation='sigmoid')

        # model = BisenetV2(detail_filters=32, aggregation_filters=32, num_class=1, final_act='sigmoid')

        model = U2Net(rsu_middle_filters=32,
                      rsu_out_filters=64,
                      num_class=1,
                      end_activation='sigmoid',
                      only_output=True)

        # model = U2Net(rsu_middle_filters=16,
        #               rsu_out_filters=32,
        #               num_class=20,
        #               end_activation='softmax',
        #               only_output=True)

        if self.learning_rate > 0:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            print('[INFO] 使用sgd,其值为：\t' + str(self.learning_rate))
        else:
            optimizer = 'Adam'
            print('[INFO] 使用Adam')

        model.compile(
            optimizer=optimizer,
            loss=dice_loss(),
            metrics=[tf.keras.metrics.Precision()]
        )
        # model.compile(
        #     optimizer=optimizer,
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=['categorical_accuracy']
        # )

        if os.path.exists(self.checkpoint_input_path + '.index') and self.load_weights:
            print("[INFO] -------------------------------------------------")
            print("[INFO] -----------------loading weights-----------------")
            print("[INFO] -------------------------------------------------")
            model.load_weights(self.checkpoint_input_path)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_save_path,
            monitor='val_precision',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            save_freq='epoch')

        history = model.fit(
            self.train_datasets,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.val_datasets,
            validation_freq=1,
            callbacks=[checkpoint_callback]
        )

        if self.epochs == 1:
            # 一般都是训练前专门看一下信息 所以正常训练时就不显示了 主要还是tmux不能上翻 有的时候会遮挡想看的信息
            model.summary()

        return history


def plot_learning_curves(history, plt_name):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig('./log/' + plt_name + '.jpg')


def train_init():
    # ex_info = 'dense_unet_df32sf16_mix_loss'
    # ex_info = 'detail_con_unet_face_edge_focal'
    # ex_info = 'bisev2_mix_loss'
    # ex_info = 'u2net_dice'
    # ex_info = 'u2net_16_64_bce_dice'
    # ex_info = 'u2net_16_64'
    # ex_info = 'u2net_mix_loss_mix_precision'
    # ex_info = 'detail_con_unet_face_edge'
    # ex_info = 'u2net_dice_02aug42000'
    # ex_info = 'u2net_bin_02aug10000'
    # ex_info = 'u2net_bin_01aug16000'
    # ex_info = 'u2net_seg'
    # ex_info = 'u2net_16_64_dice'
    # ex_info = 'unet_16_32_4_dice'
    # ex_info = 'u2net_dice_02aug42000'
    ex_info = 'u2net_32_64_dice'

    print('[INFO] 实验名称：' + ex_info)

    start_time = datetime.datetime.now()

    tran_tab = str.maketrans('- :.', '____')
    plt_name = ex_info + str(start_time).translate(tran_tab)

    args = parseArgs()
    seg = train(load_weights=args.load_weights,
                batch_size=args.batch_size,
                epochs=args.epochs,
                load_train_file_number=args.load_train_file_number,
                load_val_file_number=args.load_val_file_number,
                data_augmentation=args.data_augmentation,
                learning_rate=args.learning_rate,
                ex_info=ex_info)

    history = seg.model_train()
    plot_learning_curves(history, plt_name)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    train_init()
