'''
Downscale each image of the training and testing set using Bicubic 
'''

IMG_HR_DIR = '../DataSet/IMG_HR'
IMG_LR_DIR = '../DataSet/IMG_LR_X3'
DOWNSCALE = 2
import cv2
import os.path as osp

with open('train.txt', 'r') as f:
    for imgname in f.readlines():
        print(imgname.strip())
        img = cv2.imread(osp.join(IMG_HR_DIR, imgname.strip() + '.bmp'))
        img = cv2.resize(img, None, fx=1/DOWNSCALE, fy=1/DOWNSCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(IMG_LR_DIR, imgname.strip() + 'x%d' %DOWNSCALE + '.bmp'), img)

with open('val.txt', 'r') as f:
    for imgname in f.readlines():
        print(imgname.strip())
        img = cv2.imread(osp.join(IMG_HR_DIR, imgname.strip() + '.bmp'))
        img = cv2.resize(img, None, fx=1/DOWNSCALE, fy=1/DOWNSCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(IMG_LR_DIR, imgname.strip() + 'x%d' %DOWNSCALE + '.bmp'), img)

with open('test5.txt', 'r') as f:
    for imgname in f.readlines():
        print(imgname.strip())
        img = cv2.imread(osp.join(IMG_HR_DIR, imgname.strip() + '.bmp'))
        img = cv2.resize(img, None, fx=1/DOWNSCALE, fy=1/DOWNSCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(IMG_LR_DIR, imgname.strip() + 'x%d' %DOWNSCALE + '.bmp'), img)

with open('test14.txt', 'r') as f:
    for imgname in f.readlines():
        print(imgname.strip())
        img = cv2.imread(osp.join(IMG_HR_DIR, imgname.strip() + '.bmp'))
        img = cv2.resize(img, None, fx=1/DOWNSCALE, fy=1/DOWNSCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(IMG_LR_DIR, imgname.strip() + 'x%d' %DOWNSCALE + '.bmp'), img)
