# coding=utf-8
from glob import glob
from utils import shuffle_file, distribution_img_label, recreate_dir, distribution_file


def img_label_distribution(ori_img_file_path,
                           ori_label_file_path,
                           train_img_file_path,
                           train_label_file_path,
                           val_img_file_path,
                           val_label_file_path,
                           split_rate=0.8,
                           is_recreate_dir=False):
    ori_img_file_list = glob(ori_img_file_path + '*.jpg')
    ori_label_file_list = glob(ori_label_file_path + '*.png')

    assert len(ori_img_file_list) == len(ori_label_file_list)

    ori_img_file_list.sort()
    ori_label_file_list.sort()

    ori_img_file_list, ori_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)

    train_file_num = int(len(ori_img_file_list) * split_rate)

    print('[info] train num:\t' + str(train_file_num) + '\tval num:\t' + str(len(ori_img_file_list) - train_file_num))

    print('[info] -- -- --\t train数据分发 \t-- -- --')
    distribution_img_label(distribution_img_file_list=ori_img_file_list[:train_file_num],
                           distribution_label_file_list=ori_label_file_list[:train_file_num],
                           distribution_img_file_path=train_img_file_path,
                           distribution_label_file_path=train_label_file_path,
                           is_recreate_dir=is_recreate_dir)

    print('[info] -- -- --\t val数据分发 \t-- -- --')
    distribution_img_label(distribution_img_file_list=ori_img_file_list[train_file_num:],
                           distribution_label_file_list=ori_label_file_list[train_file_num:],
                           distribution_img_file_path=val_img_file_path,
                           distribution_label_file_path=val_label_file_path,
                           is_recreate_dir=is_recreate_dir)


def celeb_label_distribution():
    label_path = '../data/celeb_ori_label/'
    save_label_path = '../data/celeb_ori_label_temp/'
    for index in range(15):
        print(index)
        temp_label_path = label_path + str(index) + '/'
        label_file_list = glob(temp_label_path+'*.png')
        distribution_file(label_file_list, save_label_path)


def celeb_img_edge_distribution():
    img_path = '../data/celeb_ori_img/'
    edge_path = '../data/celeb_edge/'
    save_train_img_path = '../data/temp/train/img/'
    save_train_label_path = '../data/temp/train/label/'
    save_val_img_path = '../data/temp/val/img/'
    save_val_label_path = '../data/temp/val/label/'

    img_file_path_list = glob(img_path + '*.jpg')
    label_file_path_list = glob(edge_path + '*.png')
    print(len(img_file_path_list), len(label_file_path_list))

    img_file_path_list.sort()
    label_file_path_list.sort()

    img_file_path_list = img_file_path_list[:300]
    label_file_path_list = label_file_path_list[:300]

    distribution_img_label(img_file_path_list[:200], label_file_path_list[:200],
                           save_train_img_path, save_train_label_path)
    distribution_img_label(img_file_path_list[200:], label_file_path_list[200:],
                           save_val_img_path, save_val_label_path)


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

    img_label_distribution(ori_img_file_path='../data/celeb_ori_img/',
                           ori_label_file_path='../data/celeb_edge/',
                           train_img_file_path='../data/train/img/',
                           train_label_file_path='../data/train/label/',
                           val_img_file_path='../data/val/img/',
                           val_label_file_path='../data/val/label/',
                           split_rate=0.95
                           is_recreate_dir=True)

    # celeb_label_distribution()
    # celeb_img_edge_distribution()
