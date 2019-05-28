import cv2
import tensorflow as tf
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.utils.generic_utils import get_custom_objects
from utils import RGB2YCrCb, crop, psnr, normalize

IMG_HR_DIR = '../DIV2K/DIV2K_HR'
IMG_LR_DIR_2X = '../DIV2K/DIV2K_LR_bicubic_X2'
IMG_LR_DIR_3X = '../DIV2K/DIV2K_LR_bicubic_X3'

TRAIN_IDS = '../DIV2K/train.txt'
TEST_IDS = '../DIV2K/test.txt'
VAL_IDS = '../DIV2K/val.txt'

IMG_SIZE_MAX = (648, 1116) # (H,W)
IMG_SIZE = (32, 32)

# To work on RTX GPU:
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
# added because cudnn & cublas fail:
tfconfig.gpu_options.allocator_type = 'BFC'
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9 
sess = tf.Session(config=tfconfig)
keras.backend.tensorflow_backend.set_session(sess)

def split_data(train_file, test_file, val_file):
    with open(train_file, 'r') as f:
        train_ids = f.readlines()
        train_ids = [id.strip() for id in train_ids]
    with open(test_file, 'r') as f:
        test_ids = f.readlines()
        test_ids = [id.strip() for id in test_ids]
    with open(val_file, 'r') as f:
        val_ids = f.readlines()
        val_ids = [id.strip() for id in val_ids]
    return train_ids, test_ids, val_ids

def load_HR_img(id):
    imgRGB = cv2.imread(osp.join(IMG_HR_DIR, '%s.png' % id))
    # imgYCC = RGB2YCrCb(imgRGB)
    # imgYCC = imgYCC[:,:,0]
    img = normalize(imgRGB)
    return img #img[:,:, np.newaxis]

def load_LR_img(id, downscale=2):
    if downscale == 3:
        dir = IMG_LR_DIR_3X
    else:
        dir = IMG_LR_DIR_2X
    imgRGB = cv2.imread(osp.join(dir, '%sx%s.png' % (id, downscale)))
    imgRGB = cv2.resize(imgRGB, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_CUBIC)
    # imgYCC = RGB2YCrCb(imgRGB)
    # imgYCC = imgYCC[:,:,0]
    img = normalize(imgRGB)
    return img #img[:,:, np.newaxis]


class DataGenerator(keras.utils.Sequence):
    '''Data Generator that streams data to the Keras model by batches'''
    def __init__(self, list_ids, batch_size=32, dim=IMG_SIZE, 
                n_channels=1, downscale=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.downscale = downscale
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def prepare_data(self, id, downscale):
        '''Crop the image by choosing a random window of observation'''
        img_HR = load_HR_img(id)
        img_LR = load_LR_img(id, downscale)

        marginx = img_HR.shape[0] - self.dim[0]
        marginy = img_HR.shape[1] - self.dim[1]
        # define randomly the top left pixel:
        startx = np.random.randint(0, marginx)
        starty = np.random.randint(0, marginy)

        cropped_img_HR = img_HR[startx:startx+self.dim[0], \
                                starty:starty+self.dim[1], \
                                :]
        cropped_img_LR = img_LR[startx:startx+self.dim[0], \
                                starty:starty+self.dim[1], \
                                :]                          
        return cropped_img_LR, cropped_img_HR

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        #Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            X[i,], y[i,] = self.prepare_data(id, self.downscale)
            
        return X, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
       
        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_ids_temp)
        return X, y

# def weight_init(shape, dtype=tf.float32):
#     kernel = np.random.normal(0.0, 0.001, shape)
#     centerX = shape[0]//2
#     centerY = shape[1]//2
#     kernel[centerX][centerY] = 1.0
#     kernel = tf.convert_to_tensor(kernel, dtype=dtype)
#     return kernel


def build_model(params, checkpoint):
    if checkpoint:
        model = load_model(checkpoint)
    else:
        weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)
        model = Sequential()
        model.add(Conv2D(128, (9,9), padding='same', activation=tf.nn.relu, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', input_shape=(*params['dim'], params['n_channels']), name='conv1'))
        model.add(Conv2D(64, (3,3), padding='same', activation=tf.nn.relu, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', name='conv2'))
        model.add(Conv2D(params['n_channels'], (5,5), padding='same', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', name='conv3'))
    return model
    
def predict_model(params, checkpoint):
    if checkpoint:
        model = load_model(checkpoint)
    else:
        weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)
        model = Sequential()
        model.add(Conv2D(128, (9,9), padding='same', activation=tf.nn.relu, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', input_shape=(None, None, params['n_channels']), name='conv1'))
        model.add(Conv2D(64, (3,3), padding='same', activation=tf.nn.relu, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', name='conv2'))
        model.add(Conv2D(params['n_channels'], (5,5), padding='same', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', name='conv3'))
    return model

# def get_custom_optimizer():
#     LR_dict = {'conv1': 10, 'conv2': 10, 'conv3': 1} # lr multipliers
#     opt = LR_SGD(lr=1e-4, multipliers=LR_dict)
#     return opt

def train_model(model, train_gen, val_gen, save_path):
    if save_path:
        checkpoint = ModelCheckpoint(save_path,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='auto',
                                    period=40)
        callbacks_list = [checkpoint]

    model.compile(optimizer=Adam(lr=0.00003), loss='mean_squared_error')

    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  use_multiprocessing=True,
                                  workers=2,
                                  verbose=1,
                                  epochs=400,
                                  callbacks=callbacks_list)
    return history

def plot_training(history):
    # Compute psnr
    train_mse = history.history['loss']
    val_mse = history.history['val_loss']
    train_psnr = 20 * np.log10(1./np.sqrt(train_mse))
    val_psnr = 20 * np.log10(1./np.sqrt(val_mse))
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(train_psnr,'r',linewidth=3.0)
    plt.plot(val_psnr,'b',linewidth=3.0)
    plt.legend(['Training PSNR', 'Validation PSNR'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('PSNR [dB]',fontsize=16)
    plt.title('PSNR Curves',fontsize=16)
    plt.show()

if __name__ == '__main__':
    # Load img ids for each set
    train_ids, test_ids, val_ids = split_data(TRAIN_IDS, TEST_IDS, VAL_IDS)

    # Load data generators
    params = {'dim': IMG_SIZE,
              'batch_size': 64,
              'n_channels': 3,
              'downscale': 2,
              'shuffle': True}

    train_generator = DataGenerator(train_ids, **params)
    val_generator = DataGenerator(val_ids, **params)
    test_generator = DataGenerator(test_ids, **params)

    # Design model
    checkpoint = osp.join('weights', 'weights_Adam_32x32x3_RGB.120-0.00118.hdf5')
    try:
        model = build_model(params, checkpoint)
    except:
        opt = get_custom_optimizer()
        get_custom_objects().update({"LR_SGD": opt})
        model = build_model(params, checkpoint)
    model.summary()

    #Train model
    save_path = osp.join('weights', 'weights_Adam_32x32x3_RGB.{epoch:02d}-{val_loss:.5f}.hdf5')
    train_results = train_model(model, train_generator, val_generator, save_path)
    plot_training(train_results)

    # Test model
    test_results = model.evaluate_generator(test_generator)
    print("Test Loss: ", test_results)
    print('Test PSNR: ', 20*np.log10(1./np.sqrt(test_results)))

    # Save predictions
    predictions = model.predict_generator(generator=test_generator,
                                          use_multiprocessing=False,
                                          workers=1,
                                          verbose=1)
    
    np.save('predictions', predictions)
