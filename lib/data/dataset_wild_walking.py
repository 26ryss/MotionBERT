import os
import glob
import json
import pickle
import io
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import ipdb

from lib.utils.utils_data import crop_scale, resample


def halpe2h36m(x):
    '''
        Input: x (T x V x C)
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus!=None and item['idx']!=focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = halpe2h36m(kpts_all)
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)

class AlphaPoseWildDataset(Dataset):
    """
    Takes a json file path and returns the corresponding data
    self.X: list of numpy array of shape (2, n_frames, 17, 3), second person is fake
    self.y: list of labels
    """
    def __init__(self, json_path, n_frames=243, random_move=True, scale_range=[1,1]):
        # Create the json paths list and corresponding labels list
        self.json_path = json_path
        self.random_move = random_move
        self.scale_range = scale_range
        self.n_frames = n_frames
        self.X = self._process_json()
        print(f"INFO: Loaded {json_path}, shape: {self.X.shape}")

    def __len__(self):
        """Denotes the total number of samples"""
        return 1

    def _process_json(self):
        """
        Process the json file and return the corresponding data
        """
        motion = np.array(read_input(self.json_path, vid_size=None, scale_range=self.scale_range, focus=None))
        resample_id = resample(ori_len=motion.shape[0], target_len=self.n_frames, randomness=False)
        motion = motion[resample_id]
        fake = np.zeros(motion.shape)
        motion = np.array([motion, fake])
        return motion.astype(np.float32)

    def __getitem__(self, index):
        """
        Returns a sample of data
        self.X: numpy array of shape (2, n_frames, 17, 3), second person is fake
        """
        return self.X
