import os

import torch
import scipy.io

from un_utils import find_similar, gen_opt
from un_createdb import extract_feature, create_dataloaders, prepare_model

def similar_images(model, path):
    features = {}
    for dir_ in [file for file in os.listdir('static/data') if os.path.isdir(os.path.join('static/data', file))]:
        if not os.path.exists(f'static/data/{dir_}/features.mat'):
            print(f'no static/data/{dir_}/features.mat exists')
            continue
        print(f'loading features from static/data/{dir_}')
        features[dir_] = scipy.io.loadmat(f'static/data/{dir_}/features.mat')

    #features_arr = [features[key]['features'] for key in features.keys()]
    #print(len(features_arr))
    #print(len(features_arr[0]))

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
    similar = {}
    for key in features.keys():
        similar[key] = find_similar(feature, torch.FloatTensor(features[key]['features']))
    paths_to_similar = {}
    for key in similar.keys():
        paths_to_similar[key] = []
        for index in similar[key]:
            paths_to_similar[key].append(features[key]['path'][index])

    #print(paths_to_similar.keys())
    #for key in paths_to_similar.keys():
    #    print(key, paths_to_similar[key])

    return paths_to_similar

def similar_images_test(model, path):
    features = {}
    for dir_ in [file for file in os.listdir('static/data') if os.path.isdir(os.path.join('static/data', file))]:
        if not os.path.exists(f'static/data/{dir_}/test_features.mat'):
            print(f'no static/data/{dir_}/test_features.mat exists')
            continue
        print(f'loading features from static/data/{dir_}')
        features[dir_] = scipy.io.loadmat(f'static/data/{dir_}/test_features.mat')

    #features_arr = [features[key]['features'] for key in features.keys()]
    #print(len(features_arr))
    #print(len(features_arr[0]))

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
    similar = {}
    for key in features.keys():
        similar[key] = find_similar(feature, torch.FloatTensor(features[key]['features']))
    paths_to_similar = {}
    for key in similar.keys():
        paths_to_similar[key] = []
        for index in similar[key]:
            paths_to_similar[key].append(features[key]['path'][index])

    #print(paths_to_similar.keys())
    #for key in paths_to_similar.keys():
    #    print(key, paths_to_similar[key])

    return paths_to_similar

if __name__ == '__main__':
    model = prepare_model('three_view_long_share_d0.75_256_s1_google', gen_opt())
    imgs = similar_images(model, 'static/temp/0000/0001.jpg')
