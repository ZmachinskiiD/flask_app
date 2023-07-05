import os
import argparse
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
import scipy.io

from un_utils import load_network, gen_opt

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name) -> int:
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def scale(ms):
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))
    return ms

def extract_feature(model, dataloaders, opt, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    print('view_index=', view_index)
    print('opt.views=', opt.views)
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img)
            for scale_ in scale(opt.ms):
                if scale_ != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale_, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None)
                    elif view_index ==2:
                        _, outputs = model(None, input_img)
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def prepare_model(name, opt):
    model, _, epoch = load_network(name, opt)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    return model

def create_dataloaders(data_dir, opt):
    img_dirs = [file for file in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, file))]
    img_dirs = [dir_ for dir_ in img_dirs if not os.path.exists(os.path.join(data_dir, dir_, 'features.mat'))]
    print(f'creating dataloaders for the following directories in {data_dir} :', img_dirs)

    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in img_dirs}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=False, num_workers=4) for x in img_dirs}
    return dataloaders, image_datasets

#unused
def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

if __name__ == '__main__':
    opt = gen_opt()
    dataloaders, image_datasets = create_dataloaders('static/data', opt)
    model = prepare_model('three_view_long_share_d0.75_256_s1_google', opt)

    with torch.no_grad():
        for dir_name in image_datasets.keys():
            print(f'creating features for static/data/{dir_name}')
            view = which_view('drone')
            features = extract_feature(model, dataloaders[dir_name], opt, view)
            fts = []
            for i in range(len(features)):
                fts.append((image_datasets[dir_name].imgs[i][0], features[i].numpy()))
            test_result = {'imgs_fts':fts}
            result = {
                'features':features.numpy(),
                'label': view,
                'path': image_datasets[dir_name].imgs,
            }
            print(f'saving features for static/data/{dir_name}')
            scipy.io.savemat(f'static/data/{dir_name}/features.mat', result)
            scipy.io.savemat(f'static/data/{dir_name}/test_features.mat', test_result)
    print('done')
