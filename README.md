# Image Super Resolution

This code implements a Super Resolution Convolutional Neural Network (SRCNN) implemented by Chao Dong et al (see paper in docs/super_res_2.pdf)

## Folder Structure

├── code  
│   ├── downscale.py  
│   ├── model.py  
│   ├── results  
│   ├── test.py  
│   ├── utils.py  
│   └── weights  
├── DataSet  
│   ├── IMG_HR  
│   ├── IMG_LR_X2  
│   ├── IMG_LR_X3  
│   ├── test14.txt  
│   ├── test5.txt  
│   ├── train.txt  
│   └── val.txt  
├── DIV2K  
│   ├── DIV2K_HR  
│   ├── DIV2K_LR_bicubic_X2  
│   ├── DIV2K_LR_bicubic_X3  
│   ├── test.txt  
│   ├── train.txt  
│   └── val.txt  
├── docs  
│   └── super_res_2.pdf  
└── README.md  

## Remark

In this repository you will not find the folders DIV2K and DataSet which are simply composed by the image dataset that have been used to train the convolutional neural network, however you can find them both in the following links:
### DIV2K
https://data.vision.ee.ethz.ch/cvl/DIV2K/
### DataSet

## How to train
Run the model.py script and make sure to adapt the following variables to your dataset:

- IMG_HR_DIR: path to your High Resolution images
- IMG_LR_DIR_2X: path to your Low Resolution images (downsample = 2)
- IMG_LR_DIR_3X = path to your Low Resolution images (downsample = 3)

- TRAIN_IDS: text file containing the name of images used for training
- TEST_IDS: text file containing the name of images used for testing
- VAL_IDS: text file containing the name of images used for validation

- IMG_SIZE: sub-image size used for training
- DOWNSCALE: downscale factor of the Low resolution images

You can adapt the training parameters by modifying the following dict:
params = {'dim': IMG_SIZE,
          'batch_size': 32,
          'n_channels': 1,
          'downscale': DOWNSCALE,
          'shuffle': True}

## How to test
Run the test.py script and make sure to adapt the following variables to your dataset:

- IMG_HR_DIR: path to your High Resolution images
- IMG_LR_DIR_2X: path to your Low Resolution images (downsample = 2)
- IMG_LR_DIR_3X = path to your Low Resolution images (downsample = 3)

- TRAIN_IDS: text file containing the name of images used for training
- TEST_IDS: text file containing the name of images used for testing
- VAL_IDS: text file containing the name of images used for validation

- DOWNSCALE: downscale factor of the Low resolution images
- VISUALIZE: Set to True if you want to save the results on /results folder

and make sure to load your .hdf5 model into the model

