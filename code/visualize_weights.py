import numpy as np
import os
import os.path as osp
import h5py
import cv2
from utils import unnormalize

def get_hdf5_datasets_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = f'{prefix}/{key}'
        print(path)
        if isinstance(item, h5py.Dataset):  # test for Dataset
            #print('new item')
            yield (path, item)
        elif isinstance(item, h5py.Group):  # test for Group (go down)
            yield from get_hdf5_datasets_iterator(item, path)

def read_hdf5(hdf5_file_name):
    with h5py.File(hdf5_file_name, 'r') as f:
        for (path, item) in get_hdf5_datasets_iterator(f):
            yield (path, item)

def visualize_weights(hdf5_file_name, ext='.png'):
    #with h5py.File(file_name, 'r') as file:
    image_folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'weights_visualization_' + osp.splitext(osp.basename(hdf5_file_name))[0])
    if not(osp.exists(image_folder_path)):
        os.mkdir(image_folder_path)

    mat_dict = {}
    is_first_tensor = True
    for (path, item) in read_hdf5(hdf5_file_name):
        if path.startswith('/model_weights'):
            tensor = np.array(item)
            # print(tensor.shape)
            if len(tensor.shape) == 4:
                if is_first_tensor:
                    wmin = np.amin(tensor)
                    wmax = np.amax(tensor)
                    is_first_tensor = False
                    print(wmin)
                    print(wmax)
                else:
                    wmin = min(wmin, np.amin(tensor))
                    wmax = max(wmax, np.amax(tensor))
                for j in range(tensor.shape[3]):
                    for i in range(tensor.shape[2]):
                        mat = tensor[:,:,i,j]
                        #print(mat)
                        image_file_name = (path[1:].replace('/','_')).replace(':','_') + '_{}_{}'.format(i, j) + ext
                        image_path = osp.join(image_folder_path,image_file_name)
                        #print(image_path)
                        mat_dict[image_path] = mat
    print(wmin)
    print(wmax)
    for image_path in mat_dict.keys():
        mat_dict[image_path] = unnormalize(mat_dict[image_path], wmin, wmax)
        cv2.imwrite(image_path, mat_dict[image_path])

#        for key in file.keys():
#            print(key)
#            if(key == "model_weights"):
#                print("yes")
#                weights = file[key]
#                print(weights)
                #print(weights)
#                print("break")
 #               break
            #print(min(data))
            #print(max(data))
            #print(data[:15])
    return 0


if __name__ == '__main__':
    visualize_weights(osp.join('weights', 'mehdi_small_Y.2800-0.00115.hdf5'))#'weights_Adam_32x32x3_RGB.120-0.00118.hdf5' 'weights_Adam_32x32.160-0.00074.hdf5' 'mehdi_Y.2800-0.00084.hdf5'