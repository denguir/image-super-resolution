import numpy as np
import os.path as osp
import cv2
from utils import zero_padding, psnr, RGB2YCrCb, YCrCb2RGB
from model import predict_model, split_data, load_HR_img, load_LR_img

# IMG_HR_DIR = '../DIV2K/DIV2K_HR'
# IMG_LR_DIR_2X = '../DIV2K/DIV2K_LR_bicubic_X2'
# IMG_LR_DIR_3X = '../DIV2K/DIV2K_LR_bicubic_X3'

# TRAIN_IDS = '../DIV2K/train.txt'
# TEST_IDS = '../DIV2K/test.txt'
# VAL_IDS = '../DIV2K/val.txt'

IMG_HR_DIR = '../DataSet/IMG_HR'
IMG_LR_DIR_2X = '../DataSet/IMG_LR_X2'

TRAIN_IDS = '../DataSet/train.txt'
TEST_IDS = '../DataSet/test5.txt'
VAL_IDS = '../DataSet/val.txt'

DOWNSCALE = 2
VISUALIZE = True

# Load test data:
_, test_ids, _ = split_data(TRAIN_IDS, TEST_IDS, VAL_IDS)

# build network, agnostic to input size
params = {'dim': None,
          'batch_size': 1,
          'n_channels': 1,
          'downscale': DOWNSCALE,
          'shuffle': False}
model = predict_model(params, None)

# load model with a weigth file:
modelname_mehdi_RGB = 'mehdi_RGB.320-0.00131.hdf5'
modelname_mehdi_Y = 'mehdi_Y.2800-0.00084.hdf5'
modelname_Y = 'weights_Adam_32x32.160-0.00074.hdf5'
modelname_RGB = 'weights_Adam_32x32x3_RGB.120-0.00118.hdf5'
modelname_RGB2 = 'weights_Adam_32x32x3_RGB.320-0.00087.hdf5'

model.load_weights(osp.join('weights', modelname_mehdi_Y))

# Evaluate the PSNR for each image of the test set
# Set VISUALIZE = True to register the images in 'results' directory
PSNR_bicubic = []
PSNR_pred = []
for id in test_ids:
    imgHR = load_HR_img(id, folder=IMG_HR_DIR, ext='bmp')
    imgHR = imgHR[np.newaxis,:,:,:]
    imgLR = load_LR_img(id, folder=IMG_LR_DIR_2X, ext='bmp', downscale=DOWNSCALE)
    imgLR = imgLR[np.newaxis,:,:,:]
    imgPred = model.predict(imgLR)
    
    psnr_bic_id = psnr(imgLR, imgHR, 1.0)
    psnr_net_id = psnr(imgPred, imgHR, 1.0)
    print('PSNR bicubic for %s: %s' %(id, psnr_bic_id))
    print('PSNR network for %s: %s' %(id, psnr_net_id))
    PSNR_bicubic.append(psnr_bic_id)
    PSNR_pred.append(psnr_net_id)

    if VISUALIZE:
        LR = np.squeeze(imgLR) * 255
        LR.astype(np.uint8)

        HR = np.squeeze(imgHR) * 255
        HR.astype(np.uint8)

        pred = np.squeeze(imgPred) * 255
        pred.astype(np.uint8)
        pred[pred[:] > 255] = 255
        pred[pred[:] < 0] = 0

        cv2.imwrite(osp.join('results', '%s_LR.png' %id), LR)
        cv2.imwrite(osp.join('results','%s_HR.png' %id), HR)
        cv2.imwrite(osp.join('results', '%s_pred.png' %id), pred)

print('PSNR bicubic Total: ', sum(PSNR_bicubic)/len(PSNR_bicubic))
print('PSNR network Total: ', sum(PSNR_pred)/len(PSNR_pred))
