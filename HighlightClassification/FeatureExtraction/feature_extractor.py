import logging
import os
from tqdm import tqdm
import torch
import numpy as np


def feature_extractor(dataloader, model, feature_path, model_name, split, device):

    logging.info("start training")

    best_loss = 9e99
    
    features, groundtruths = [], []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            labels = labels.to(torch.int)
            # measure data loading time
            feats = feats.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feature = model(feats, return_feats=True)

            feature_dir_x = os.path.join(feature_path, split[0], 'x')
            feature_dir_y = os.path.join(feature_path, split[0], 'y')
            os.makedirs(feature_dir_x, exist_ok=True)
            os.makedirs(feature_dir_y, exist_ok=True)
            with open(os.path.join(feature_dir_x,'batch'+str(i)+'.npy'), 'wb') as f:
                np.save(f, feature.cpu().numpy())
            with open(os.path.join(feature_dir_y,'batch'+str(i)+'.npy'), 'wb') as f:
                np.save(f, labels.cpu().numpy())   

    return 

def feature_extractor_test(dataloader, model, feature_path, model_name, split, device):

    logging.info("start training")

    best_loss = 9e99

    features = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, feats in t:
            # measure data loading time
            feats = feats.to(device)

            with torch.no_grad():
                feature = model(feats, return_feats=True)
            
            features.append(feature.cpu().numpy())
        
    features = np.concatenate(features, axis=0)

    feature_dir = feature_path + model_name
    os.makedirs(feature_dir, exist_ok=True)
    with open(feature_dir+'/'+ split[0] + '.npy', 'wb') as f:
        np.save(f, features)   

    return feature