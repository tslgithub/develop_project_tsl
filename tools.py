#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
# Date:    2018.04
# Copyright © 2018 Xuexiyou Company. All rights reserved.
######################################################################

import sys
import cv2
import  os
import  re
import numpy as np
from matplotlib import pyplot as plt
import inspect

def sh(data):
    plt.imshow(data)
    plt.show()

def sh2(data):
    dpi = 80.0
    xpixels, ypixels = data.shape[::-1]
    margin = 0.05
    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    ax.imshow(data, interpolation='none')
    plt.show()

def box(width, height):
    return np.ones((height, width), dtype=np.uint8)

'''

'''
def debug_pic(name,data):
    cv2.imwrite(name+".png", data)

def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def contact_contours(contours):
    (cnts, boundingBoxes) = sort_contours(contours)
    lines = list([])
    last_y = -999
    for contour in boundingBoxes:
        x, y, w, h = contour
        if abs(y - last_y) > 20:
            lines.append([x, y, w, h])
        else:
            min_x = min(lines[-1][0], x)
            max_x = max(lines[-1][0] + lines[-1][2], x + w)
            lines[-1][0] = min_x
            lines[-1][2] = max_x - min_x
        last_y = y
    return lines

def ocr(mat):
    cv2.imwrite("_ocr_h.png", mat)
    os.system("tesseract" + " " + "_ocr_h.png" + " ocr  -l chi_sim --psm 7")
    fs = open("ocr.txt", 'r', encoding='utf-8')
    string = fs.read()
    string = re.sub(r'\n', "", string)
    print("ocr: " + string)
    return string

def ocr_number(mat):
    cv2.imwrite("_ocr_h.png", mat)
    os.system("tesseract" + " " + "_ocr_h.png" + " ocr --oem 0 --psm 7 test_c")
    fs = open("ocr.txt", 'r', encoding='utf-8')
    string = fs.read()
    string = re.sub(r'\D', "", string)
    print("ocr: " + string)
    return string

def find_next_page(ck_y,contours_with_page):
    for contour in contours_with_page:
        x, y, w, h, page = contour
        if(ck_y > (y+20) ):
            continue
        return page
    return contours_with_page[-1][-1]

def use_max_location(x, y, v, locations):
    if (len(locations)==0):
        locations.append([x, y, v])
        return
    for loc in locations:
        ck_x, ck_y, ck_v = loc
        if( abs(ck_x-x)<20 and abs(ck_y-y)<20):
            if(ck_v<v):
                locations[-1]=[x,y,v]
            return
    locations.append([x, y, v])

def get_pic_height(mat):
    locs = np.where(mat > 200)[0]
    return locs.max() - locs.min() + 1

    mask = cv2.dilate(mat, box(10, 1))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_height = 0
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        max_height = max(max_height, h)
    return max_height

def change_height_with_template(mat,tf):
    mat_height = get_pic_height(mat) #return max_height
    tf_height = get_pic_height(tf)
    if(mat_height==0):
        return 1, mat
    w, h = mat.shape[::-1]
    power_y = tf_height/mat_height
    change_mat = cv2.resize(mat, (int(round(w*power_y)), int(round(h*power_y))), interpolation=cv2.INTER_CUBIC)
    return power_y, change_mat

def find_with_template(mat,tf2):
    threshold = 0.75
    # need change height to the same
    power_y, change_mat = change_height_with_template(mat,tf2)
    res1 = cv2.matchTemplate(change_mat, tf2, cv2.TM_CCORR_NORMED)
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res1)
    loc = np.where(res1 >= threshold)
    fix_loc = []
    for pt in zip(*loc[::-1]):
        use_max_location(pt[0], pt[1], res1[pt[1]][pt[0]], fix_loc)
    # need revert x,y location
    revert_loc = []
    while len(fix_loc)>0:
        v = fix_loc.pop()
        v[0] = int(round(v[0] / power_y))
        v[1] = int(round(v[1] / power_y))
        revert_loc.insert(0,v)
    return revert_loc,power_y

def check_contour_in_locations(contour, locations, power_y, tf):
    x, y, w, h = contour
    ck_w, ck_h = round(tf.shape[::-1][0]/power_y), round(tf.shape[::-1][1]/power_y)
    for loc in locations:
        ck_x, ck_y, _ = loc
        if x>=ck_x and y>=ck_y and (x+w)<(ck_x+ck_w) and (y+h)<(ck_y+ck_h):
            return  loc

def check_parentheses(mat):
    left_tf = cv2.imread("xxy/parentheses_left.jpg")
    left_tf = cv2.cvtColor(left_tf, cv2.COLOR_RGB2GRAY)
    left_loc, power_left_y = find_with_template(mat, left_tf)
    right_tf = cv2.imread("xxy/parentheses_right.jpg")
    right_tf = cv2.cvtColor(right_tf, cv2.COLOR_RGB2GRAY)
    right_loc, power_right_y = find_with_template(mat, right_tf)
    if len(left_loc)>0 and len(right_loc)>0:
        _, contours, _ = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cnts, boundingBoxes) = sort_contours(contours, "left-to-right")
        for contour in boundingBoxes:
            loc = check_contour_in_locations(contour, left_loc, power_left_y, left_tf)
            loc = check_contour_in_locations(contour, right_loc, power_left_y, right_tf)


    return left_loc, right_loc, power_left_y, power_right_y

def check_is_empty(mat):
    _, contours, _ = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    v=2

class DD():
    def __init__(self, n="__"):
        # _p 最后一次的Mat值， n 图片主名 f 处理方法名 l1 第一层 l2 第二层
        self._p, self.n, self.f, self.l1, self.l2 = None, n, "_", 0, 0

    @property
    def p1(self):
        return self._p

    @p1.setter
    def p1(self, p):
        self._p = p
        self.l1 += 1
        if self.l2 > 0:
            self.f = ""
        self.l2 = 0
        self.write()

    @p1.deleter
    def p1(self):
        del self._p

    @property
    def p2(self):
        return self._p

    @p2.setter
    def p2(self, p):
        self._p = p
        self.l2 += 1
        self.write()

    @p2.deleter
    def p2(self):
        del self._p

    def write(self):
        #return
        cv2.imwrite("tmp/%s.z.%s.%s.%s.png" % (self.n, str(self.l1).zfill(2), str(self.l2).zfill(2), self.f), self._p)

