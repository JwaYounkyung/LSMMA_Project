import logging
import os
from tqdm import tqdm
import torch
import numpy as np


def feature_extractor(dataloader, model, model_name, split, device):

    logging.info("start training")

    best_loss = 9e99

    features, groundtruths = [], []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            feats = feats.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feature = model(feats, return_feats=True)
            
            features.append(feature.cpu().numpy())
            groundtruths.append(labels.cpu().numpy())
        
    features = np.concatenate(features, axis=0)
    groundtruths = np.concatenate(groundtruths, axis=0)

    feature_dir = 'data/model_features/' + model_name
    os.makedirs(feature_dir, exist_ok=True)
    with open(feature_dir+'/'+ split[0] + '.npy', 'wb') as f:
        np.save(f, features)   
    with open(feature_dir+'/'+ split[0] + '.label.npy', 'wb') as f:
        np.save(f, groundtruths)   

    return feature, groundtruths

def feature_extractor_test(dataloader, model, model_name, split, device):

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

    feature_dir = 'data/model_features/'+model_name
    os.makedirs(feature_dir, exist_ok=True)
    with open(feature_dir+'/'+ split[0] + '.npy', 'wb') as f:
        np.save(f, features)   

    return feature