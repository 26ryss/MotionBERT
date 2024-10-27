import os
import os.path as osp
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import time
import random
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *

from lib.data.dataset_walking_annot import read_input
from lib.model.model_walking_rnn import ActionNet, DecoderRNN
from lib.utils.utils_data import crop_scale, resample

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/walking_rnn/MB_ft_walking_rnn.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/walking_rnn/', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('-p', '--pretrained', default='checkpoint/pretrain/MB_release', type=str, metavar='PATH', help='pretrained checkpoint directory')
    opts = parser.parse_args()

    return opts

opts = parse_args()
args = get_config(opts.config)

vocab = read_pkl(os.path.join(opts.checkpoint, 'vocab.pkl'))
vocab_size = len(vocab)
model_backbone = load_backbone(args)

model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
decoder = DecoderRNN(embed_size=args.hidden_dim, hidden_size=512, vocab_size=vocab_size, num_layers=1)

if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()
    decoder = decoder.cuda()

model.load_state_dict(torch.load(os.path.join(opts.checkpoint, 'best_model.pth'), map_location=lambda storage, loc: storage), strict=True)
decoder.load_state_dict(torch.load(os.path.join(opts.checkpoint, 'best_decoder.pth'), map_location=lambda storage, loc: storage))

model.eval()
decoder.eval()

motion = read_input(opts.json_path, vid_size=None, scale_range=[1,1], focus=None)
motion = torch.from_numpy(motion)
resample_id = resample(ori_len=motion.shape[0], target_len=243, randomness=False)
motion = motion[resample_id]
motion = motion.unsqueeze(0)
fake = torch.zeros(1, 243, 17, 3)
motion = torch.cat([motion, fake], dim=0)
motion = motion.unsqueeze(0)

if torch.cuda.is_available():
    motion = motion.cuda()

with torch.no_grad():
    features = model(motion)
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    print("Sampled caption:\n", " ".join(sampled_caption))
