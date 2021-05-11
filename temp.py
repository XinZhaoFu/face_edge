# coding=utf-8
import cv2
import numpy as np
import datetime
from glob import glob


np.set_printoptions(threshold=np.inf)

start_time = datetime.datetime.now()

tran_tab = str.maketrans('- :.', '____')
start_time = str(start_time).translate(tran_tab)
print(start_time)

# intab = "aeiou"
# outtab = "12345"
# trantab = maketrans(intab, outtab)
#
# str = "this is string example....wow!!!";
# print
# str.translate(trantab);