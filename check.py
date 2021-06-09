from tensorflow.python.client import device_lib
from glob import glob


train_img_path = './data/train/img/'
train_label_path = './data/train/label/'
val_img_path = './data/val/img/'
val_label_path = './data/val/label/'

celeb_img_path = './data/celeb_ori_img/'
celeb_aug_img_path = './data/celeb_aug_img/'
celeb_label_path = './data/celeb_edge/'
celeb_contour_path = './data/celeb_eye_contour/'
celeb_106points_path = './data/celeb_106points/'

lapa_img_path = './data/lapa_ori_img/'
lapa_label_path = './data/lapa_edge/'
lapa_contour_path = './data/lapa_eye_contour/'
lapa_106points_path = './data/lapa_106points/'

train_img_file_list = glob(train_img_path + '*.jpg')
train_label_file_list = glob(train_label_path + '*.png')
val_img_file_list = glob(val_img_path + '*.jpg')
val_label_file_list = glob(val_label_path + '*.png')

celeb_img_file_list = glob(celeb_img_path + '*.jpg')
celeb_aug_img_file_list = glob(celeb_aug_img_path + '*.jpg')
celeb_label_file_list = glob(celeb_label_path + '*.png')
celeb_contour_file_list = glob(celeb_contour_path + '*.txt')
celeb_106points_file_list = glob(celeb_106points_path + '*.txt')

lapa_img_file_list = glob(lapa_img_path + '*.jpg')
lapa_label_file_list = glob(lapa_label_path + '*.png')
lapa_contour_file_list = glob(lapa_contour_path + '*.txt')
lapa_106points_file_list = glob(lapa_106points_path + '*.txt')

print('设备信息:')
print(device_lib.list_local_devices())

print('训练集图片数量:\t' + str(len(train_img_file_list)))
print('训练集标签数量:\t' + str(len(train_label_file_list)))
print('验证集图片数量:\t' + str(len(val_img_file_list)))
print('验证集标签数量:\t' + str(len(val_label_file_list)))
print('\n')
print('celeb图片数量:\t' + str(len(celeb_img_file_list)))
print('celeb增强图片数量:\t' + str(len(celeb_aug_img_file_list)))
print('celeb标签数量:\t' + str(len(celeb_label_file_list)))
print('celeb虹膜点数量:\t' + str(len(celeb_contour_file_list)))
print('celeb106点数量:\t' + str(len(celeb_106points_file_list)))
print('\n')
print('lapa图片数量:\t' + str(len(lapa_img_file_list)))
print('lapa标签数量:\t' + str(len(lapa_label_file_list)))
print('lapa虹膜点数量:\t' + str(len(lapa_contour_file_list)))
print('lapa106点数量:\t' + str(len(lapa_106points_file_list)))

