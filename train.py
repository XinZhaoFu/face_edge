# coding=utf-8
import argparse
import datetime
import tensorflow as tf
import os
from model.unet import UNet
from data_utils.dataloader import Data_Loader_File
import matplotlib.pyplot as plt
import pandas as pd
import setproctitle
from loss.loss import binary_focal_loss, dice_loss


gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

setproctitle.setproctitle("face_edge")


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
                        help='data_augmentation type is boolean',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    return args


class train:
    def __init__(self,
                 load_weights=False,
                 batch_size=8,
                 epochs=0,
                 load_train_file_number=0,
                 load_val_file_number=0,
                 data_augmentation=False,
                 augmentation_rate=1,
                 erase_rate=0.1,
                 learning_rate=0,
                 train_file_path='./data/train/',
                 val_file_path='./data/val/',
                 checkpoint_save_path='./checkpoint/detail_con_unet_face_edge_focal.ckpt'):
        self.load_weights = load_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.load_train_file_number = load_train_file_number
        self.load_val_file_number = load_val_file_number
        self.erase_rate = erase_rate
        self.augmentation_rate = augmentation_rate
        self.learning_rate = learning_rate
        self.checkpoint_save_path = checkpoint_save_path

        self.strategy = tf.distribute.MirroredStrategy()
        print('目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))

        data_loader = Data_Loader_File(data_augmentation=self.data_augmentation,
                                       batch_size=self.batch_size,
                                       train_file_path=train_file_path,
                                       val_file_path=val_file_path)
        self.train_datasets = data_loader.load_train_data(load_file_number=self.load_train_file_number)
        self.val_datasets = data_loader.load_val_data(load_file_number=self.load_val_file_number)

    def model_train(self):
        """
        可多卡训练
        :return:
        """
        with self.strategy.scope():
            model = UNet(semantic_filters=16,
                         detail_filters=32,
                         num_class=1,
                         semantic_num_cbr=1,
                         detail_num_cbr=6,
                         end_activation='sigmoid')

            if self.learning_rate > 0:
                print('使用sgd,其值为：\t' + str(self.learning_rate))
                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy']
                )
            else:
                print('使用adam')
                model.compile(
                    optimizer='Adam',
                    loss=binary_focal_loss(),
                    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )

            if os.path.exists(self.checkpoint_save_path + '.index') and self.load_weights:
                print("[INFO] -------------------------------------------------")
                print("[INFO] -----------------loading weights-----------------")
                print("[INFO] -------------------------------------------------")
                model.load_weights(self.checkpoint_save_path)

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_save_path,
                monitor='val_loss',
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
            callbacks=[checkpoint_callback])

        model.summary()

        return history


def plot_learning_curves(history, plt_name):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig('./log/'+plt_name+'.jpg')


def train_init():
    ex_info = '_focal_loss'
    start_time = datetime.datetime.now()

    tran_tab = str.maketrans('- :.', '____')
    plt_name = str(start_time).translate(tran_tab) + ex_info

    args = parseArgs()
    seg = train(load_weights=args.load_weights,
                batch_size=args.batch_size,
                epochs=args.epochs,
                load_train_file_number=args.load_train_file_number,
                load_val_file_number=args.load_val_file_number,
                data_augmentation=args.data_augmentation,
                learning_rate=args.learning_rate)
    history = seg.model_train()
    plot_learning_curves(history, plt_name)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    train_init()
