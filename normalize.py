#! /usr/bin/env python

import cv2
import numpy as np
import glob
import os

data, names = [], []
row, col, chan = 326, 490, 3

output_dir = 'data/normalized_images/'
mean_file, std_file = 'results/mean_arr.bin', 'results/std_arr.bin'
new_label_file = 'data/normalize_labels.txt'
OUT = open(new_label_file, 'w')
IN_list = open('data/labels.txt', 'r').readlines()
for i in range(len(IN_list)):
    line = IN_list[i]
    tmp_list = line.split(' ')
    img_name, center_x, center_y = tmp_list[0], tmp_list[1], tmp_list[2]
    img_name = 'data/raw_images/' + img_name
    img = cv2.imread(img_name)
    bname = os.path.basename(img_name)
    id = bname[:-4]
    new_path = output_dir + id + '_norm.jpg'
    names.append(new_path)
    OUT.write(new_path+' '+center_x+' '+center_y)
    data.append(list(img.flatten() ) )

OUT.close()
n = len(data)
data = np.array(data)
mean_arr, std_arr = np.mean(data, 0), np.std(data, 0)
mean_arr.astype('float').tofile(mean_file)
std_arr.astype('float').tofile(std_file)

for i in range(n):
    tmp_mean, tmp_std = mean_arr[i], std_arr[i]
    img = np.divide(data[i,:] - mean_arr, std_arr).reshape(326, 490, 3)
    cv2.imwrite(names[i], img)

