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
from lib.data.dataset_walking_annot_tranformer import AlphaPoseAnnotDataset
from lib.model.model_walking_transformer import EncoderDecoder, TokenDrop

from transformers import AutoTokenizer

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
    opts = parser.parse_args()
    return opts

def train_transformer(args, opts):
    print("INFO: Training Transformer")
    json_paths, captions = get_json_paths_and_caption('data/walking')
    train_json_paths, train_captions, test_json_paths, test_captions = split_dataset_labels_kcv(json_paths, captions, 8, 1)
    print("INFO: Loaded json paths and captions, total of {} samples".format(len(json_paths)))
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("INFO: Loaded tokenizer, total of {} words". format(len(tokenizer)))
    # dataset
    train_dataset = AlphaPoseAnnotDataset(train_json_paths, train_captions, train=True, n_frames=243, random_move=True, scale_range=[1,1])
    test_dataset = AlphaPoseAnnotDataset(test_json_paths, test_captions, train=False, n_frames=243, random_move=True, scale_range=[1,1])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    print("INFO: Loaded dataset, loading model")
    # model, decoder
    model_backbone = load_backbone(args)
    for param in model_backbone.parameters():
        param.requires_grad = False

    model = EncoderDecoder(model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, enc_hidden_dim=args.hidden_dim, num_joints=args.num_joints, vocab_size=len(tokenizer), num_layers=args.dec_num_layers, num_heads=args.dec_num_heads)
    # print number of parameters
    print("INFO: Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print trainable parameters
    print("INFO: Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    td = TokenDrop(0.05)
    test_td = TokenDrop(0.0)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    best_test_loss = 1000

    train_loss, test_loss = [], []

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        model.train()
        epoch_train_loss = 0
        epoch_test_loss = 0
        for i, (motion, captions) in enumerate(train_dataloader):
            tokens = tokenizer(captions, padding=True, return_tensors="pt", truncation=True)
            token_ids = tokens['input_ids']
            padding_mask = tokens['attention_mask']
            bs = token_ids.shape[0]

            target_ids = torch.cat((token_ids[:,1:], torch.zeros(bs, 1).long()), dim=1)
            tokens_in = td(token_ids)
            if torch.cuda.is_available():
                motion = motion.cuda()
                padding_mask = padding_mask.cuda()
                tokens_in = tokens_in.cuda()
                target_ids = target_ids.cuda()

            with torch.cuda.amp.autocast():
                pred = model(motion, tokens_in, padding_mask)
            loss = (loss_fn(pred.transpose(1,2), target_ids) * padding_mask).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, i, len(train_dataloader), loss.item()))
            epoch_train_loss += loss.item()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'epoch_{}.pth'.format(epoch)))
        train_loss.append(epoch_train_loss / len(train_dataloader))

        model.eval()
        with torch.no_grad():
            for i, (motion, captions) in enumerate(test_dataloader):
                tokens = tokenizer(captions, padding=True, return_tensors="pt", truncation=True)
                token_ids = tokens['input_ids']
                padding_mask = tokens['attention_mask']
                bs = token_ids.shape[0]

                target_ids = torch.cat((token_ids[:,1:], torch.zeros(bs, 1).long()), dim=1)
                tokens_in = test_td(token_ids)

                if torch.cuda.is_available():
                    motion = motion.cuda()
                    padding_mask = padding_mask.cuda()
                    tokens_in = tokens_in.cuda()
                    target_ids = target_ids.cuda()

                with torch.cuda.amp.autocast():
                    pred = model(motion, tokens_in, padding_mask)
                loss = (loss_fn(pred.transpose(1,2), target_ids) * padding_mask).mean()
                epoch_test_loss += loss.item()
            test_loss.append(epoch_test_loss / len(test_dataloader))

        print(f"After epoch {epoch}, losses are train: {train_loss[-1]}, test: {test_loss[-1]}")

        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            print("INFO: Saving best model")
            torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'best_model.pth'))

    display_train_test_results('vis', 'model_transformer', train_loss, train_loss, test_loss, test_loss)

    print('Finished training')

    print("INFO: Saving model")
    # save vocab, model, decoder
    if not os.path.exists(opts.checkpoint):
        os.makedirs(opts.checkpoint, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(opts.checkpoint, 'last_epoch_model.pth'))

    return

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_transformer(args, opts)
