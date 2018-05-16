# -*- encoding --utf -8
import cv2
import numpy as np
import PIL as pil
import tools
import os
import json
import matplotlib as plt
from pylab import *
#from skimage import data, util, measure, color
#from skimage.measure import label
import math
import re

# PROJECT_PATH = "/home/xxy/project/tsl_project/"
# path1="/home/xxy/project/tsl_project/"
# path2="/home/xxy/project/tsl_project/tmp/"
# path3="/home/xxy/project/tsl_project/image/"


def sh(mat):
    mat=cv2.resize(mat,(0,0),None,0.2,0.2)
    cv2.imshow("mat",mat)
    cv2.waitKey()

def sh2(data):
    dpi = 80.0
    xpixels, ypixels = data.shape[::-1]
    margin = 0.05
    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    ax.imshow(data, interpolation='none')
    plt.show()

def sh3(data):
    plt.imshow(data)
    plt.show()

def dp(d, mat):
    dot = cv2.imread("/dot.7.7.jpg")
    dirs = []
    dir = []

    d.p2=img_gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    w, h = img_gray.shape[::-1]

    l = get_content_roi(d, img_gray)  #去边框
    # img_struct = img_gray[l[0]:l[1],l[2]:l[3]]
    img_struct = img_gray[slice(l[0], l[1]), slice(l[2], l[3])]
    loc = find_mid_line(d, img_struct)   #切割线,原函数是img2part
    half_img =cut_image(d,img_struct,loc)     #切分图形,  返回是图形的左右图，在list中

    for half in  half_img:
        d.p2=no_circle = find_circles(d,half,dir)   #返回的e二值化的图
        dirs=dp_split_height2(d,no_circle,dir)
        break
    # d.p2 = img_gray = img_gray[slice(0,h),slice(0,w)]
    # d.p1 = content = get_content_roi(d,img_gray)   #返回去边框后的图
    # get_content_roi(d, img_gray)

    dirs=sort_content(dir,dirs)
    ocr_test(dirs)

       #返回图像的中分线的位置x
    # img_part = read_img_path(content,loc)  #返回左右图像的切分后的列表，   返回list,    list[0]、list[1]分别为左右图
    # dirs = []
    # dir = []
    # for image in img_part:
    #     no_circle = find_circles(d, image, dir)
    #     img_content = dp_split_height2(d, no_circle, dir)  # 返回的图像，只剩下二级目录
    #     find_content2(d, img_content, dir)
    #     sort_content(dir, dirs)
    #     ocr_test(dirs)


def get_content_roi(d,mat):
    copy_mat = mat
    d.f = "get_content_roi"
    d.p2 = img = cv2.adaptiveThreshold(copy_mat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 25)
    w, h = mat.shape[::-1]
    w4, h4 = int(w/4), int(h/4)

    d.p2 = top_roi = img[slice(0, h4), slice(w4, 3*w4)]
    ret,top_thresh = cv2.threshold(top_roi,0,255,cv2.THRESH_OTSU)
    d.p2 = mask = cv2.dilate(top_thresh, tools.box(150,80))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, boundingBoxes = tools.sort_contours(contours)
    top = boundingBoxes[0][3]

    d.p2 = left_roi = img[slice(h4, 3*h4), slice(0, w4)]
    ret, left_thresh = cv2.threshold(left_roi, 0, 255, cv2.THRESH_OTSU)
    d.p2 = mask = cv2.dilate(left_thresh, tools.box(100, 10))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, boundingBoxes = tools.sort_contours(contours, method="left-right")
    left = boundingBoxes[0][2]

    d.p2 = right_roi = img[slice(h4, 3*h4), slice(3*w4, w)]
    ret, right_thresh = cv2.threshold(right_roi, 0, 255, cv2.THRESH_OTSU)
    d.p2 = mask = cv2.dilate(right_thresh, tools.box(100, 10))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, boundingBoxes = tools.sort_contours(contours, method="left-right")
    right = 3*w4+boundingBoxes[-1][0]

    d.p2 = bottom_roi = img[slice(3*h4, h), slice(w4, 3*w4)]
    ret, botton_thresh = cv2.threshold(bottom_roi, 0, 255, cv2.THRESH_OTSU)
    d.p2 = mask = cv2.dilate(botton_thresh, tools.box(2, 100))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, boundingBoxes = tools.sort_contours(contours)
    bottom = 3*h4+boundingBoxes[-1][1]

    d.p2 = content = copy_mat[slice(top,bottom), slice(left,right)]
    # return content
    # return [slice(top,bottom), slice(left,right)]
    return [top,bottom,left,right]

def find_mid_line(d, img):
    mat=img
    w,h =mat.shape[::-1]
    d.p2 = mat = mat[:,int(0.4*w):int(0.6*w)]

    minLineLength = int(h / 2)
    maxLineGap = 15

    d.p2 = mat_adaptive = cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 15)

    d.p2 = mask = cv2.dilate(mat_adaptive, tools.box(3, 10))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 30, minLineLength, maxLineGap)

    for x1, y1, x2, y2 in lines[0]:
        print("x1=", x1, "; y1=", y1, "; x2=", x2, "; y2=", y2)
        d.p2 = drae_line = cv2.line(mat,(x1,y1),(x2,y2),(255,255,0),10)

    loc = int(0.5*( x1 + x2 )+0.4*w)
    # # cv2.imwrite(path3+d.n+"_left.jpg",orimat[:,0:loc])
    # # cv2.imwrite(path3+d.n+"_right.jpg",orimat[:,loc:w])
    return loc

def cut_image(d,img,loc):
    w, h = img.shape[::-1]
    d.p2=img_left = img[:,0:loc]
    d.p2=img_right = img[:,loc:w]
    return [img_left,img_right]

def find_circles(d,img,dir):
    img_gray = img
    width,hight = img.shape[::-1]
    # d.p2=img
    d.p2=roi_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 275, 15)
    d.p2=roi_adaptive_weak = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 25)
    # cv2.GaussianBlur(img_gray,img_gray,tools.box(2,2),2)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=130, maxRadius=160)
    cx ,cy,cr= [],[],[]
    ContentPath1,Height1,PagePath1,ClassPath =[],[],[],[]
    k = 1
    for circle in circles[0,:]:

        # print("i = ", i)
        x,y,r = circle[0],circle[1],circle[2]
        cx.append(x)
        cy.append(y)
        cr.append(r)
        # d.p2 =cir_demo=  cv2.circle(img, (x, y),r, (0, 0, 255),5 )
        # cv2.circle(img, (x, y), r, (0, 0, 255), 1)
        a,b,c,dd = int(y - r+20),int(y),int(x - r / 2),int(x + r / 2)
        print (x," ",y,"   ",r)

        d.p2 = roi = roi_adaptive[a:b,c:dd] #一目录的数字
        d.p2 = content = roi_adaptive_weak[slice(int(y-r),int(y+r)),slice(int(x+r+40),width)]#一级目录的文字
        # ocr_number(roi)
        # ocr(content)

        content_path="./image/"+d.n+"-content-1-"+str(k)+".jpg"
        ContentPath1.append(content_path)
        cv2.imwrite(content_path, content)

        Height1.append(y-r)

        roi_path="./image/"+d.n+"-page-1-"+str(k)+".jpg"
        PagePath1.append(0)
        ClassPath.append(roi_path)
        cv2.imwrite(roi_path,roi)
        k+=1
        roi_adaptive_weak[slice(int(y - r-30), int(y + r+20)), slice(0, width)]=0

        dir.extend(list(zip(ContentPath1, Height1, PagePath1,ClassPath)))

    return  roi_adaptive_weak

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

def dp_split_height2(d,roi,dir):#roi已经在find_circle中自适应二值化
    # roi_copy = roi
    # roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 25)
    mask = cv2.dilate(roi, tools.box(2, 2))
    tf = cv2.imread("./dot.7.7.jpg")
    tf2 = cv2.cvtColor(tf, cv2.COLOR_RGB2GRAY)
    #template = cv2.adaptiveThreshold(tf2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 25)
    res = cv2.matchTemplate(roi, tf2, cv2.TM_CCORR_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    regions = np.zeros_like(roi)
    # found all match dot image point
    for pt in zip(*loc[::-1]):
        cv2.rectangle(regions, pt, (pt[0] + 10, pt[1] + 5), (255, 255, 0), 2)
    d.p2 = regions
    regions = cv2.dilate(regions, tools.box(5, 5))

    _, contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = tools.contact_contours(contours)
    regions = np.zeros_like(roi)
    i = 0
    width, height = tf2.shape[::-1]
    for contour in lines:
        x, y, w, h = contour
        contour[2]=contour[2]+width-10
        contour[3] = height+2
    for contour in lines:
        x, y, w, h = contour
        cv2.rectangle(regions, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # cv2.putText(regions, "{}".format(i + 1), (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        i += 1
    d.p2 = regions
    cv2.imwrite("./dot_match.png",regions)
    # ret,roi_for_contents = cv2.threshold(roi_copy,100,255,cv2.THRESH_OTSU)
    content = contents(d,roi,lines,dir)
    return content

def contents(d,roi,lines,dir):#roi已经在find_circle中自适应二值化
    roi_cpy = roi
    # roi = cv2.imread("/home/xxy/project/tsl_project/dir1_right.jpg",0)
    width,hight = roi.shape[::-1]
    kk=0
    content_height = []
    ContentPath3,Height3,PagePath3 ,ClassPath= [],[],[],[]

    for contour in lines:
        x,y,w,h = contour
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        d.p2=img1=content_target = roi[slice(y,h+y+20),slice(0,x)]
        d.p2=img2=page_target = roi[slice(y,h+y+20),slice(x+w-5,width-50)]
        content_height.extend([y])
        for i in content_height:
            pass
            # print ("content_height = ",i)
        # print ("均值 = ",mean(content_target))
        if mean(content_target)<10:
            content_target = roi[slice(y-h-30,y+10),slice(60,width)]
        elif len(content_height)>1:
            height = content_height[kk]-content_height[kk-1]
            # print("高的差值为： ",height)
            if height>1.65*h and height<2.5*h:
                img1=roi[slice(y, h + y + 20), slice(0, x)]
                img2=roi[slice(y-h-20, y ), slice(0, x)]
                img=img_concatenate(img1,img2)
                sh3(img)
        content_target = cv2.erode(content_target, tools.box(2,2))

        page_target = cv2.erode(page_target, tools.box(2,2))
        # content_target = img_concatenate(img1, img2)
        cv2.imwrite("./image/" + d.n + "-content-3-" + str(kk) + ".jpg", content_target)

        cv2.imwrite("/image"+d.n+"_class3_pageNUM_"+str(kk)+".jpg",page_target)
        PagePath3.append("./image/"+d.n+"_class3_pageNUM_"+str(kk)+".jpg")
        ContentPath3.append("./image/"+d.n+"_class3_content_"+str(kk)+".jpg")
        Height3.append(h+y+10)
        ClassPath.append(0)

        # PagePath3.append(path3+d.n+"_class3_pageNUM_"+str(kk)+".jpg")

        roi[slice(y , y+h+10), slice(0, width)]=0
        roi[slice(y - h - 30, y + 10), slice(0, width)] = 0

        kk+=1
    dir.extend(list(zip(ContentPath3,Height3,PagePath3,ClassPath)))
    # CONTENT.append(list(zip(ContentPath3,Height3,PagePath3)))
    d.p2 = roi
    return dir

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

def read_target(dir):
    k=1
    f= open("./image/target.txt","w+")
    for i in dir:
        for j in i:
            print(i,k)
            f.write(str(j))
            f.write(" ")
            k+=1
        f.write("\n")
    f.close

def sort_content(dir,dirs):
    sort_dir = sorted(dir, key=lambda b: b[1])
    dirs.extend(sort_dir)
    read_target(dirs)
    # ocr_test(dirs)
    return dirs

def ocr_test(dirs):
    f = open("./image/"+"ocr_detection.txt","w")
    for imgpath in dirs:
        img = cv2.imread(imgpath[0],0)
        ocr_content = tools.ocr(img)
        img_num_path = imgpath[2]
        if img_num_path != 0:
            # continue
            img_num = cv2.imread(img_num_path,0)
            # img_num = cv2.cvtColor(img_n)
            ocr_num = tools.ocr_number(img_num)
        else: ocr_num ='0'
    # for i in range(len(dirs)):
    #     img = cv2.imread(dirs[i][0],0)
    #     img2= cv2.imread(dirs[i][2],0)
    #     ocr_content = tools.ocr(img)
    #     if dirs[2] !=0 or dirs[2] != "0":
    #         ocr_NUM = tools.ocr_number(img2)
    #     else:
    #         ocr_NUM = "0"
    #     f.write(ocr_content +"\n")
        f.write(ocr_content+"   "+ocr_num)
        f.write("\n")

def img_concatenate(img1,img2):
    # img1=cv2.imread("/home/ccs-pc/project/dir1.jpg",0)
    # img2=cv2.imread("/home/ccs-pc/project/dir2.jpg",0)
    img = np.concatenate([img1,img2],axis=1)

    # image = np.concatenate((gray1, gray2))  # 纵向连接=np.vstack((gray1, gray2))
    # 横向连接image = np.concatenate([gray1, gray2], axis=1)

    return img

def main():
    imgfiles = ["Dir.jpg"]
    for imgfile in imgfiles:
        basename = os.path.basename(imgfile)
        name, _ = os.path.splitext(basename)
        img = cv2.imread(imgfile)
        d = tools.DD(name)
        dp(d, img)

if __name__ == '__main__':
    main()