# coding=utf-8
from glob import glob
from utils import shuffle_file, distribution_img_label, recreate_dir, distribution_file


def img_label_distribution(ori_img_file_path='../data/lapa_ori_img/',
                           ori_label_file_path='../data/edge/',
                           train_img_file_path='../data/train/img/',
                           train_label_file_path='../data/train/label/',
                           val_img_file_path='../data/val/img/',
                           val_label_file_path='../data/val/label/'
                           ):
    ori_img_file_list = glob(ori_img_file_path + '*.jpg')
    ori_label_file_list = glob(ori_label_file_path + '*.png')

    assert len(ori_img_file_list) == len(ori_label_file_list)

    ori_img_file_list.sort()
    ori_label_file_list.sort()

    ori_img_file_list, ori_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)

    train_file_num = int(len(ori_img_file_list) * 0.8)

    print('train num:\t' + str(train_file_num) + '\t\tval num:\t' + str(len(ori_img_file_list) - train_file_num))

    recreate_dir(train_img_file_path)
    recreate_dir(train_label_file_path)

    print('[info] -- -- --\t train数据分发 \t-- -- --')
    distribution_img_label(dis_img_file_list=ori_img_file_list[:train_file_num],
                           dis_label_file_list=ori_label_file_list[:train_file_num],
                           dis_img_file_path=train_img_file_path,
                           dis_label_file_path=train_label_file_path)

    print('[info] -- -- --\t val数据分发 \t-- -- --')
    distribution_img_label(dis_img_file_list=ori_img_file_list[train_file_num:],
                           dis_label_file_list=ori_label_file_list[train_file_num:],
                           dis_img_file_path=val_img_file_path,
                           dis_label_file_path=val_label_file_path)


def celeb_label_distribution():
    label_path = '../data/celeb_ori_label/'
    save_label_path = '../data/celeb_ori_label_temp/'
    for index in range(15):
        print(index)
        temp_label_path = label_path + str(index) + '/'
        label_file_list = glob(temp_label_path+'*.png')
        distribution_file(label_file_list, save_label_path)


if __name__ == '__main__':
    """
    都是复制操作不是移动操作
    """
    # distribution()

    # distribution(ori_img_file_path='../data/lapa_ori_img/',
    #              ori_label_file_path='../data/edge/',
    #              train_img_file_path='../data/train/img/',
    #              train_label_file_path='../data/train/label/',
    #              val_img_file_path='../data/val/img/',
    #              val_label_file_path='../data/val/label/')

    # img_label_distribution(ori_img_file_path='../data/lapa_ori_img/',
    #                        ori_label_file_path='../data/lapa_nose_label/',
    #                        train_img_file_path='../data/nose_train/img/',
    #                        train_label_file_path='../data/nose_train/label/',
    #                        val_img_file_path='../data/nose_val/img/',
    #                        val_label_file_path='../data/nose_val/label/')

    celeb_label_distribution()

