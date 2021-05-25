from tensorflow.python.client import device_lib
from glob import glob


train_img_path = './data/train/img/'
train_label_path = './data/train/label/'
val_img_path = './data/val/img/'
val_label_path = './data/val/label/'

train_img_file_list = glob(train_img_path + '*.jpg')
train_label_file_list = glob(train_label_path + '*.png')
val_img_file_list = glob(val_img_path + '*.jpg')
val_label_file_list = glob(val_label_path + '*.png')

print('设备信息:')
print(device_lib.list_local_devices())

print('训练集图片数量:\t' + str(len(train_img_file_list)))
print('训练集标签数量:\t' + str(len(train_label_file_list)))
print('验证集图片数量:\t' + str(len(val_img_file_list)))
print('验证集标签数量:\t' + str(len(val_label_file_list)))

