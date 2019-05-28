import numpy as np
import os.path as osp
import cv2
from utils import zero_padding, psnr, RGB2YCrCb, YCrCb2RGB
from model2 import predict_model, split_data, DataGenerator

IMG_HR_DIR = '../DIV2K/DIV2K_HR'
IMG_LR_DIR_2X = '../DIV2K/DIV2K_LR_bicubic_X2'
IMG_LR_DIR_3X = '../DIV2K/DIV2K_LR_bicubic_X3'

TRAIN_IDS = '../DIV2K/train.txt'
TEST_IDS = '../DIV2K/test.txt'
VAL_IDS = '../DIV2K/val.txt'

IMG_SIZE_MAX = (648, 1116) # (H,W)
IMG_SIZE = (510, 510)
DOWNSCALE = 2

# Load test data:
_, test_ids, _ = split_data(TRAIN_IDS, TEST_IDS, VAL_IDS)

params = {'dim': IMG_SIZE,
          'batch_size': 10,
          'n_channels': 3,
          'downscale': DOWNSCALE,
          'shuffle': False}

test_generator = DataGenerator(test_ids, **params)

PSNR_bicubic = []
PSNR_pred = []
model = predict_model(params, None)
modelname_Y = 'weights_Adam_32x32.160-0.00074.hdf5'
modelname_RGB = 'weights_Adam_32x32x3_RGB.120-0.00118.hdf5'
model.load_weights(osp.join('weights', 'weights_Adam_32x32x3_RGB.320-0.00087.hdf5'))


for i in range(10):
    imgLR, imgHR = test_generator.__getitem__(i)
    # load model:
    imgHR_pred = model.predict(imgLR, batch_size=params['batch_size'])

    PSNR_bicubic.append(psnr(imgLR, imgHR, 1.0))
    PSNR_pred.append(psnr(imgHR_pred, imgHR, 1.0))

print('PSNR bicubic: ', sum(PSNR_bicubic)/len(PSNR_bicubic))
print('PSNR network: ', sum(PSNR_pred)/len(PSNR_pred))

# Visualize one example

imgLR, imgHR = test_generator.__getitem__(0)
pred = model.predict(imgLR, batch_size=params['batch_size'])

for i in range(params['batch_size']):
    LR = imgLR[i,:,:,:] * 255
    LR.astype(np.uint8)
    
    HR = imgHR[i,:,:,:] * 255
    HR.astype(np.uint8)
    
    predi = pred[i,:,:,:] * 255
    predi.astype(np.uint8)
    predi[predi[:] > 255] = 255
    predi[predi[:] < 0] = 0

    print('psnr bicubic example %d' %i, psnr(HR, LR, 255))
    print('psnr network example %d' %i, psnr(HR, predi, 255))

    cv2.imwrite(osp.join('results', 'ex%d_LR.png' %i), LR)
    cv2.imwrite(osp.join('results','ex%d_HR.png' %i), HR)
    cv2.imwrite(osp.join('results', 'ex%d_pred.png' %i), predi)