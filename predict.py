# coding=utf-8
import glob
import cv2
from model.unet import UNet
from model.bisenetv2 import BisenetV2
import tensorflow as tf
import numpy as np
import datetime

np.set_printoptions(threshold=np.inf)

start_time = datetime.datetime.now()

# checkpoint_save_path = './checkpoint/face_edge.ckpt'
checkpoint_save_path = './checkpoint/detail_con_unet_face_edge.ckpt'
# checkpoint_save_path = './checkpoint/bisev2_3x2_face_edge.ckpt.index'
predict_save_path = './data/predict/'
test_file_path = './data/test/'

# 加载模型
print('[info]加载模型')
# model = BisenetV2(detail_filters=32, aggregation_filters=32, final_filters=2)
model = UNet(filters=32, num_class=2, semantic_num_cbr=1, detail_num_cbr=4)
model.compile(optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.load_weights(checkpoint_save_path)

print('[info]加载图像 开始计时')
start_time = datetime.datetime.now()

test_img = cv2.imread('./data/test/demo1.jpg')
test_img = cv2.resize(test_img, dsize=(512, 512))
test_img = test_img / 255.
test_img_np = np.empty(shape=(1, 512, 512, 3), dtype=np.float32)

test_img_np[0:, :, :, :] = test_img

print('[info]执行推理')
predict_temp = model.predict(test_img_np)
print(predict_temp.shape)

predict_img = np.empty(shape=(512, 512), dtype=np.uint8)
predict_img[:, :] = predict_temp[0, :, :, 1] * 255
predict_img = cv2.blur(predict_img, ksize=(5, 5))
predict_img = cv2.adaptiveThreshold(predict_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
# _, predict_img = cv2.threshold(predict_img, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite('./data/predict/detail_con_unet_demo1.png', predict_img)

end_time = datetime.datetime.now()
print('time:\t' + str(end_time - start_time).split('.')[0])


# print(predict_temp[0, 256, :, :])
#
# # 获取预测图的文件名
# test_label_list = glob.glob(test_label_file_path + '*.png')
# img_name_list = []
# for img_file in test_label_list:
#     img_name = (img_file.split('\\')[-1])
#     img_name_list.append(img_name)
#
# num_test_list = []
# for _ in range(13):
#     num_test_list.append(0)
#
# predict_img_list = []
# for predict in predict_list:
#     predict_img = np.ones(shape=(256, 256, 3), dtype=np.uint8)
#     predict_img *= 128
#
#     for row in range(256):
#         for col in range(256):
#             predict_img[row][col][2] = predict[row][col] * 10
#             num_test_list[predict[row][col]] += 1
#
#     predict_img_list.append(predict_img)
#
# for predict_img, img_name in zip(predict_img_list, img_name_list):
#     cv2.imwrite(predict_save_path + img_name, predict_img)
#
# print(num_test_list)
