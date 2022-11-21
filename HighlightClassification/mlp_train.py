# 아직 편집 다 안함
#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("--list_videos", default="data/labels/train_val.csv")
parser.add_argument("--video_feat_name", default="cnn") #

parser.add_argument("--output_file", default="weights/earlyfusion.mlp.model") 
parser.add_argument("--model_name", default="NetVLAD++_PCA512") 


if __name__ == '__main__':

  args = parser.parse_args()
  # fread = open(args.list_videos, "r")
  
  # load video names and events in dict
  # df_videos_label = {}
  # for line in open(args.list_videos).readlines()[1:]:
  #   video_id, category = line.strip().split(",")
  #   df_videos_label[video_id] = category

# %% Ensemble Features

  fusion_list = []
  with open('data/model_features/' + args.model_name + '/train.npy', 'rb') as f:
    audio_feat_list = np.load(f)
  with open('data/model_features/' + args.model_name + '/valid.npy', 'rb') as f:
    video_feat_list = np.load(f)
  with open('features/labels.npy', 'rb') as f:
    label_list = np.load(f)

  for i in range(len(audio_feat_list)):
    fusion = np.concatenate([audio_feat_list[i], video_feat_list[i]])
    fusion_list.append(fusion)

# %% Run MLP
  #1. Train a MLP classifier using feat_list and label_list
  # below are the initial settings you could use
  # hidden_layer_sizes=(512),activation="relu",solcer="adam",alpha=1e-3
  # your model should be named as "clf" to match the variable in pickle.dump()
  clf = MLPClassifier(hidden_layer_sizes=(1024),activation="relu",solver="adam",alpha=1e-4, verbose=True)
  clf.fit(fusion_list, label_list)

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
