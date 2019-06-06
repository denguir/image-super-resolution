import numpy as np
import os.path as osp
import cv2
from utils import zero_padding, psnr, RGB2YCrCb, YCrCb2RGB
from model import predict_model, split_data, load_HR_img, load_LR_img

# Dataset n°1:

IMG_HR_DIR = '../DIV2K/DIV2K_HR'
IMG_LR_DIR_2X = '../DIV2K/DIV2K_LR_bicubic_X2'
IMG_LR_DIR_3X = '../DIV2K/DIV2K_LR_bicubic_X3'

TRAIN_IDS = '../DIV2K/train.txt'
TEST_IDS = '../DIV2K/test.txt'
VAL_IDS = '../DIV2K/val.txt'

# Dataset n°2:

# IMG_HR_DIR = '../DataSet/IMG_HR'
# IMG_LR_DIR_2X = '../DataSet/IMG_LR_X2'
# IMG_LR_DIR_3X = '../DataSet/IMG_LR_X3'

# TRAIN_IDS = '../DataSet/train.txt'
# TEST_IDS = '../DataSet/test5.txt'
# VAL_IDS = '../DataSet/val.txt'

# REMAIDER:
# When you choose a Downscale factor, make sure to use the appropriate directory
# for low resolution images:
# DOWNSCALE = 2 -> Use IMG_LR_DIR_2X folder in load_LR_img
# DOWNSCALE = 3 ->  Use IMG_LR_DIR_2X folder in load_LR_img
DOWNSCALE = 2 
VISUALIZE = False

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
modelname_mehdi_Y = 'mehdi_Y.2800-0.00084.hdf5' # arch 2
modelname_mehdi_Y_div2k = 'mehdi_Y.40-0.00075.hdf5' # arch 2
modelname_mehdi_Y_small = 'mehdi_small_Y.2800-0.00115.hdf5' # arch 1

model.load_weights(osp.join('weights', modelname_mehdi_Y_div2k))

# Evaluate the PSNR for each image of the test set
# Set VISUALIZE = True to register the images in 'results' directory
PSNR_bicubic = []
PSNR_pred = []
for id in test_ids:
    try:
        imgHR = load_HR_img(id, folder=IMG_HR_DIR, ext='png')
        imgHR = imgHR[np.newaxis,:,:,:]
        imgLR = load_LR_img(id, folder=IMG_LR_DIR_2X, ext='png', downscale=DOWNSCALE)
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
    except ValueError:
        print('Image %s failed probably because the downscaling-upsclaing bicubic didnt keep the original dimension\n\
               Make sur the dimension of the original image can be devided by %s' %(id, DOWNSCALE))

print('PSNR bicubic Total: ', sum(PSNR_bicubic)/len(PSNR_bicubic))
print('PSNR network Total: ', sum(PSNR_pred)/len(PSNR_pred))
