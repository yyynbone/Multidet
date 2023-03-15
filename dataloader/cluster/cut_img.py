# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:39:26 2022

@author: ZJ
"""

import cv2
import numpy as np
import math


x_size, y_size = 13152, 4384  # x_size相当于列,y_size相当于行
cut_size, stride = 832, 640

x_num = math.ceil((x_size-cut_size)/stride) + 1
y_num = math.ceil((y_size-cut_size)/stride) + 1
image = cv2.imread('D:/newdir2/CCD/CCD_0.tiff')

print(x_num, y_num)
count = 0
for j in range(y_num):
    y_start = j*stride
    x_start = 0
    for i in range(x_num):
        x_start = i*stride
        y_end = y_start + cut_size
        x_end = x_start + cut_size
        if y_end > y_size:
            y2 = y_size
        else:
            y2 = y_end
        if x_end > x_size:
            x2 = x_size
        else:
            x2 = x_end
        tmp_img = np.zeros((cut_size, cut_size, 3))
        tmp_img[:y2-y_start, :x2-x_start] = image[y_start:y2, x_start:x2]
        cv2.imwrite('D:/newdir2/aaa/test_{}.jpg'.format(count), tmp_img)
        count += 1
print('finish')
            
