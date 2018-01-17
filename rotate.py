#! /usr/bin/env python

import cv2
import numpy as np


def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotateImage(image, angle, center_x, center_y):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    M_rot = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat = np.vstack([M_rot, [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    point = np.array([[center_x, center_y, 1.]])
    point_new = np.asarray(M_rot.dot(point.T).T)
    point_new = [point_new[0][0]+dx, point_new[0][1]+dy]

    length_new, width_new, chan_new = result.shape
    point_new = [point_new[0]/width_new, point_new[1]/length_new]
    result = cv2.resize(img, (326,490), interpolation = cv2.INTER_LINEAR)

    return result, point_new[0], point_new[1]


IN_list = open('data/labels.txt', 'r').readlines()
rotate_folder = 'data/rotated_images/'
label_file = 'data/labels.txt'
labels_new = 'data/rotate_labels.txt'
label_out = open(labels_new, 'w')
label_list = open(label_file, 'r').readlines()
for line in label_list:
    label_out.write('data/raw_images/'+line)

k = 0
n_copy = 10
for line in IN_list:
    tmp_list = line.split(' ')
    img_name, center_x, center_y = tmp_list[0], float(tmp_list[1]), float(tmp_list[2])
    img_name = 'data/raw_images/' + img_name
    angles = np.random.random_integers(0, 360, n_copy)
    img = cv2.imread(img_name)
    length, width, chan = img.shape
    center_x, center_y = width*center_x, length*center_y
    for i in range(n_copy):
        img_new, x_new, y_new = rotateImage(img, angles[i], center_x, center_y)
        tmp_new_path = rotate_folder + img_name[:-4] + '_rotate_' + str(angles[i]) + '.jpg'
        cv2.imwrite(tmp_new_path, img_new)
        if k == len(IN_list)-1 and i == 19:
            label_out.write(tmp_new_path+' '+str(x_new)+' '+str(y_new))
        else:
            label_out.write(tmp_new_path+' '+str(x_new)+' '+str(y_new)+'\n')

label_out.close()
