import os
import scipy.io
import torch
import numpy as np

features = {}
for dir_ in [file for file in os.listdir('static/data') if os.path.isdir(os.path.join('static/data', file))]:
    if not os.path.exists(f'static/data/{dir_}/test_features.mat'):
        print(f'no static/data/{dir_}/test_features.mat exists')
        continue
    print(f'loading features from static/data/{dir_}')
    features[dir_] = scipy.io.loadmat(f'static/data/{dir_}/test_features.mat')


imgs_fts = [features[dir_name]['imgs_fts'][i] for dir_name in features.keys() for i in range(len(features[dir_name]['imgs_fts']))]
fts = torch.FloatTensor(np.array([imgs_fts[i][1] for i in range(len(imgs_fts))]))
print(fts.shape)
fts = fts.squeeze()
print(fts.shape)

