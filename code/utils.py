import cv2
import numpy as np
import os


def RGB2YCrCb(img):
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return imgYCC

def YCrCb2RGB(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return imgRGB

def psnr(img1, img2, maxVal):
    imdiff = img1 - img2
    rmse = np.sqrt(np.mean(np.square(imdiff)))
    if rmse > 0:
        psnr = 20 * np.log10(maxVal/rmse)
    else:
        psnr = 'inf'
    return psnr

def zero_padding(img, target_size):
    '''Pad the image with zeros to match target_size'''
    pad_img = np.zeros(target_size)
    h, w = img.shape[0], img.shape[1]
    pad_img[:h, :w] = img
    return pad_img

def crop(img, target_size):
    '''Crop the image by choosing a random window of observation'''
    marginx = img.shape[0] - target_size[0]
    marginy = img.shape[1] - target_size[1]
    # define top left pixel:
    startx = np.random.randint(0, marginx)
    starty = np.random.randint(0, marginy)

    cropped_img = img[startx:startx+target_size[0], \
                      starty:starty+target_size[1], \
                      :]
    return cropped_img

def normalize(img):
    img = img/255.0
    return img


