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

from transformers import AutoTokenizer

from lib.utils.tools import *
from lib.utils.learning import *

from lib.data.dataset_walking_annot import read_input
from lib.model.model_walking_transformer import EncoderDecoder
from lib.utils.utils_data import crop_scale, resample

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/walking_transformer/MB_ft_walking_transformer.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/walking_transformer/', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('-p', '--pretrained', default='checkpoint/pretrain/MB_release', type=str, metavar='PATH', help='pretrained checkpoint directory')
    opts = parser.parse_args()

    return opts

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = EncoderDecoder(model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, enc_hidden_dim=args.hidden_dim, num_joints=args.num_joints, vocab_size=len(tokenizer), num_layers=args.dec_num_layers, num_heads=args.dec_num_heads)
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()

state_dict = torch.load(os.path.join(opts.checkpoint, 'epoch_130.pth'))
model.load_state_dict(state_dict, strict=True)
print("INFO: Loaded model")
model.eval()

motion = read_input(opts.json_path, vid_size=None, scale_range=[1,1], focus=None)
motion = torch.from_numpy(motion)
resample_id = resample(ori_len=motion.shape[0], target_len=243, randomness=False)
motion = motion[resample_id]
motion = motion.unsqueeze(0)
fake = torch.zeros(1, 243, 17, 3)
motion = torch.cat([motion, fake], dim=0)
motion = motion.unsqueeze(0)
print("INFO: Loaded motion")
max_len = 30
sos_token = 101
eos_token = 102

if torch.cuda.is_available():
    motion = motion.cuda()

with torch.no_grad():
    print("INFO: Generating caption")
    caption = [sos_token]
    for _ in range(max_len):
        tokens = torch.tensor(caption).unsqueeze(0)
        padding_mask = torch.ones_like(tokens)
        pred = model(motion, tokens, padding_mask)
        pred_token = pred.argmax(-1).squeeze(0)[-1].item()
        caption.append(pred_token)
        if pred_token == eos_token:
            break

print(caption)
print(tokenizer.decode(caption))
