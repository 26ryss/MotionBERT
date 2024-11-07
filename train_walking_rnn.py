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
    train_json_paths, train_captions, test_json_paths, test_captions = split_dataset_labels_kcv(json_paths, captions, 8, 1)
    print("INFO: Loaded json paths and captions, total of {} samples".format(len(json_paths)))
    # vocab
    vocab = build_vocab(captions, threshold=3)
    vocab_size = len(vocab)
    print("INFO: Loaded vocab, total of {} words".format(vocab_size))
    # dataset
    train_dataset = AlphaPoseAnnotDataset(train_json_paths, train_captions, vocab, train=True, n_frames=243, random_move=True, scale_range=[1,1])
    test_dataset = AlphaPoseAnnotDataset(test_json_paths, test_captions, vocab, train=False, n_frames=243, random_move=True, scale_range=[1,1])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print("INFO: Loaded dataset, loading model")
    # model, decoder
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    decoder = DecoderRNN(embed_size=args.hidden_dim, hidden_size=512, vocab_size=vocab_size, num_layers=1)

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr_head, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()

    best_test_loss = 1000

    train_loss, test_loss = [], []

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        model.train()
        decoder.train()
        epoch_train_loss = 0
        epoch_test_loss = 0
        for i, (motion, captions, lengths) in enumerate(train_dataloader):
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, i, len(train_dataloader), loss.item()))
            epoch_train_loss += loss.item()
        train_loss.append(epoch_train_loss / len(train_dataloader))

        model.eval()
        decoder.eval()
        for i, (motion, captions, lengths) in enumerate(test_dataloader):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            if torch.cuda.is_available():
                motion = motion.cuda()
                captions = captions.cuda()
                targets = targets.cuda()
            features = model(motion)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            epoch_test_loss += loss.item()
        test_loss.append(epoch_test_loss / len(test_dataloader))

        print(f"After epoch {epoch}, losses are train: {train_loss[-1]}, test: {test_loss[-1]}")

        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            print("INFO: Saving best model")
            torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'best_model.pth'))
            torch.save(decoder.state_dict(), os.path.join(opts.checkpoint, 'best_decoder.pth'))

        if epoch % 10 == 0:
            print("INFO: Saving model")
            torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'epoch_{}.pth'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(opts.checkpoint, 'decoder_epoch_{}.pth'.format(epoch)))

    display_train_test_results('vis', 'model_rnn', train_loss, train_loss, test_loss, test_loss)

    print('Finished training')

    print("INFO: Saving model")
    # save vocab, model, decoder
    if not os.path.exists(opts.checkpoint):
        os.makedirs(opts.checkpoint, exist_ok=True)
    save_vocab(vocab, os.path.join(opts.checkpoint, 'vocab.pkl'))
    torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'last_epoch_model.pth'))
    torch.save(decoder.state_dict(), os.path.join(opts.checkpoint, 'last_epoch_decoder.pth'))

    return

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_rnn(args, opts)
