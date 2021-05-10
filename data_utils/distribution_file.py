# coding=utf-8
from glob import glob
from utils import shuffle_file, distribution_file, recreate_dir


def distribution(ori_img_file_path='../data/lapa_ori_img/',
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
    distribution_file(dis_img_file_list=ori_img_file_list[:train_file_num],
                      dis_label_file_list=ori_label_file_list[:train_file_num],
                      dis_img_file_path=train_img_file_path,
                      dis_label_file_path=train_label_file_path)

    print('[info] -- -- --\t val数据分发 \t-- -- --')
    distribution_file(dis_img_file_list=ori_img_file_list[train_file_num:],
                      dis_label_file_list=ori_label_file_list[train_file_num:],
                      dis_img_file_path=val_img_file_path,
                      dis_label_file_path=val_label_file_path)


if __name__ == '__main__':
    # distribution()
    distribution(ori_img_file_path='../data/lapa_ori_img/',
                 ori_label_file_path='../data/edge/',
                 train_img_file_path='../data/train/img/',
                 train_label_file_path='../data/train/label/',
                 val_img_file_path='../data/val/img/',
                 val_label_file_path='../data/val/label/')
