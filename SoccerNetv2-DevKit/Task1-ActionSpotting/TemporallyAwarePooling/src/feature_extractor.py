import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

def feature_extractor(dataloader, model, model_name, split, device):

    logging.info("start training")

    best_loss = 9e99

    features = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            feats = feats.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feature = model(feats, return_feats=True)
            
            features.append(feature.cpu().numpy())
        
    features = np.concatenate(features, axis=0)

    feature_dir = 'data/model_features/' + model_name
    os.makedirs(feature_dir, exist_ok=True)
    with open(feature_dir+'/'+ split[0] + '.npy', 'wb') as f:
        np.save(f, features)   

    return feature

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


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          device,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.to(device)
            labels = labels.to(device)
            # compute output
            output = model(feats)

            # hand written NLL criterion
            loss = criterion(labels, output)

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg
