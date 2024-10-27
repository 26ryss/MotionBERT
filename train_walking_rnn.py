import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_walking_annot import AlphaPoseAnnotDataset, build_vocab, custom_collate_fn
from lib.model.model_walking_rnn import ActionNet, DecoderRNN

from torch.nn.utils.rnn import pack_padded_sequence

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/walking/MB_ft_walking.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/walking', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint/pretrain/MB_release', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=1)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--kcv', default=False, type=bool, metavar='BOOL', help='k-fold cross validation')
    opts = parser.parse_args()
    return opts

def train_rnn(args, opts):
    print("INFO: Training RNN")
    json_paths, captions = get_json_paths_and_caption('data/walking')
    print("INFO: Loaded json paths and captions, total of {} samples".format(len(json_paths)))
    # vocab
    vocab = build_vocab(captions, threshold=2)
    vocab_size = len(vocab)
    print("INFO: Loaded vocab, total of {} words".format(vocab_size))
    # dataset
    dataset = AlphaPoseAnnotDataset(json_paths, captions, vocab, train=True, n_frames=243, random_move=True, scale_range=[1,1])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    print("INFO: Loaded dataset, loading model")
    # model, decoder
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    decoder = DecoderRNN(embed_size=args.hidden_dim, hidden_size=2048, vocab_size=vocab_size, num_layers=5)

    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr_head)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        for i, (motion, captions, lengths) in enumerate(dataloader):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            if torch.cuda.is_available():
                motion = motion.cuda()
                captions = captions.cuda()
                targets = targets.cuda()
            features = model(motion)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, i, len(dataloader), loss.item()))
    print('Finished training')
    # print("Testing with a sample")
    # test_json = 'data/walking/model/json/7.json'
    # test_data = read_input(test_json, vid_size=None, scale_range=[1,1], focus=None)
    # test_data = test_data.astype(np.float32)
    # resample_id = resample(ori_len=test_data.shape[0], target_len=243, randomness=False)
    # test_data = test_data[resample_id]
    # fake = np.zeros(test_data.shape)
    # test_data = np.array([[test_data, fake]]).astype(np.float32)
    # sampled_ids = decoder.sample(model(torch.tensor(test_data).cuda()))
    # sampled_ids = sampled_ids[0].cpu().numpy()
    # sampled_caption = []
    # for word_id in sampled_ids:
    #     word = vocab.idx2word[word_id]
    #     sampled_caption.append(word)
    #     if word == '<end>':
    #         break
    # sentence = ' '.join(sampled_caption)
    # print(sentence)

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_rnn(args, opts)
