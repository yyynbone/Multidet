# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:13:57 2022

@author: ZJ
"""
import cv2
import numpy as np
import math


x_size, y_size = 13152, 4384  # x_size相当于列,y_size相当于行
cut_size, stride = 832, 640

x_num = math.ceil((x_size-cut_size)/stride) + 1
y_num = math.ceil((y_size-cut_size)/stride) + 1
pred_f = np.loadtxt('E:/result/clas_result_CCD_{}.txt'.format(0))
pred_f = pred_f[pred_f[:,0].argsort()]
gt_f = np.loadtxt('E:/CCD/CCD_clasgt_{}.txt'.format(0))
image = cv2.imread('E:/CCD/CCD_0.tiff')

print(x_num, y_num)
count = 0
for j in range(y_num):
    y_start = j*stride
    x_start = 0
    for i in range(x_num):
        x_start = i*stride
        y_end = y_start + cut_size
        x_end = x_start + cut_size
        if pred_f[count,1] == 1:
            cv2.rectangle(image,
                          (int(x_start)+5, int(y_start)+5),
                          (int(x_end)-5, int(y_end-5)),
                          (0, 0, 255), 
                          thickness=8,
                          lineType=4)  # red
        if gt_f[count,1] == 1:
            cv2.rectangle(image,
                          (int(x_start), int(y_start)),
                          (int(x_end), int(y_end)),
                          (0, 255, 0), 
                          thickness=4,
                          lineType=4)  # red
        count += 1
            
cv2.imwrite('E:/result/test_0.jpg', image)
            
        
        
