import os

import torch
import scipy.io
import numpy as np

from un_utils import find_similar, gen_opt
from un_createdb import extract_feature, create_dataloaders, prepare_model

def similar_images(model, path):
    features_mat = {}
    for dir_ in [file for file in os.listdir('static/data') if os.path.isdir(os.path.join('static/data', file))]:
        if not os.path.exists(f'static/data/{dir_}/features.mat'):
            print(f'no static/data/{dir_}/features.mat exists')
            continue
        print(f'loading features from static/data/{dir_}')
        features_mat[dir_] = scipy.io.loadmat(f'static/data/{dir_}/features.mat')

    features_arr = [(features_mat[key]['imgs_fts'][i][0][0], features_mat[key]['imgs_fts'][i][1]) for key in features_mat.keys() for i in range(len(features_mat[key]['imgs_fts']))]

    opt = gen_opt()
    #img_name = path[path.rfind('/')+1:path.rfind('.')]
    #extension = path[path.rfind('.'):]
    #dir_ = os.path.join('static', 'temp', img_name)
    #os.mkdir(dir_)
    #copy_to = os.path.join(dir_, img_name+extension)
    #print(copy_to)
    #shutil.copyfile(path, copy_to)
    dataloader, image_dataset = create_dataloaders('static', opt)
    feature = extract_feature(model, dataloader['temp'], opt)
    similar = find_similar(feature, torch.FloatTensor(np.array([ft[1] for ft in features_arr])).squeeze())

    return [features_arr[index][0] for index in similar]

if __name__ == '__main__':
    model = prepare_model('three_view_long_share_d0.75_256_s1_google', gen_opt())
    imgs = similar_images(model, 'static/temp/0000/0001.jpg')
    print(imgs)
