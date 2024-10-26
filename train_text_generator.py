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
from torch.nn.utils.rnn import pack_padded_sequence

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_vocab import build_vocab
from lib.model.loss import *
from lib.data.dataset_alphapose_annotation import AlphaPoseAnnotDataset
from lib.model.model_text_generator import TextGenerator

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

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)
            acc = binary_accuracy(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, acc=accs))
    return losses.avg, accs.avg


def train_with_config_kcv(args, opts):
    json_paths = [
        'data/walking/model/json/1.json',
        'data/walking/model/json/2.json',
        'data/walking/model/json/3.json',
        'data/walking/model/json/4.json',
        'data/walking/model/json/5.json',
    ]*10
    captions = [
        "Model walks confidently with a fierce gaze and hands on hips.",
        "Model makes eye contact with the audience and gives a slight nod.",
        "Model struts with shoulders back, radiating poise and elegance.",
        "Model turns to the side and glances over their shoulder with a smirk.",
        "Model gives a playful wink to the camera while walking forward.",
        "Model holds the edge of their outfit, showing off intricate details.",
        "Model pauses mid-walk, placing one hand on the hip and tilting the head.",
        "Model twirls gracefully, showcasing the outfit's movement.",
        "Model looks up with a serene expression, embodying calmness.",
        "Model waves gently at the audience before resuming the walk.",
        "Model holds a prop and presents it while smiling at the crowd.",
        "Model gives a quick spin, allowing the fabric to flow.",
        "Model strides with purpose, making direct eye contact with the camera.",
        "Model slows down, placing one foot slightly forward for a dramatic pause.",
        "Model places both hands on the waist and poses assertively.",
        "Model raises one hand to blow a kiss toward the camera.",
        "Model does a half turn, flashing a smile back at the audience.",
        "Model stops to adjust an accessory, maintaining an elegant posture.",
        "Model points one foot forward, creating a sense of motion.",
        "Model moves hands behind the back, adding a sophisticated touch.",
        "Model stops to sway from side to side, giving a full view of the outfit.",
        "Model raises eyebrows and gives a confident, playful smile.",
        "Model pauses, gazing off into the distance with a soft expression.",
        "Model gives a peace sign toward the audience before continuing the walk.",
        "Model gently touches their necklace, drawing attention to the jewelry.",
        "Model tilts the head slightly and runs fingers through their hair.",
        "Model turns both palms outward, expressing openness.",
        "Model shifts weight onto one leg, leaning slightly to add attitude.",
        "Model tilts chin up, exuding confidence and strength.",
        "Model brings one hand up to cover their mouth in a coy gesture.",
        "Model winks playfully at a fellow model on the runway.",
        "Model walks with arms crossed, adding a touch of defiance.",
        "Model holds skirt edges delicately, lifting them slightly while walking.",
        "Model smiles warmly while glancing side to side at the audience.",
        "Model spins while moving forward, creating a flowing motion.",
        "Model bows slightly, acknowledging the crowd.",
        "Model waves excitedly, bringing energy to the walk.",
        "Model walks with a hand casually placed in a pocket.",
        "Model gestures to an accessory, drawing attention to its details.",
        "Model walks with exaggerated steps, emphasizing the outfit's fit.",
        "Model stops at the runway's end, placing one foot forward assertively.",
        "Model lifts an arm in a victorious gesture, radiating pride.",
        "Model clasps hands together and gives a big, genuine smile.",
        "Model brings one finger to lips, signaling playful mystery.",
        "Model moves with a slow, deliberate strut, giving a relaxed vibe.",
        "Model gives a quick two-finger salute to the audience.",
        "Model lifts one shoulder while looking back with a sly grin.",
        "Model mimics a snap, adding a touch of flair.",
        "Model strikes a quick pose with arms stretched out, showing the outfit's cut.",
        "Model places a hand on the chin and tilts head, lost in thought."
    ]
    print('INFO: Training with k-fold cross validation')
    all_json_paths, labels = get_data(os.path.join('data', 'walking'))
    kcv_results = {}
    # set k-cv
    k = 5
    for i in range(k):
        print(f"INFO: Training fold {i} out of {k} folds")
        train_json_paths, train_labels, test_json_paths, test_labels = split_dataset_labels_kcv(all_json_paths, labels, k, i)
        print(args)
        try:
            os.makedirs(os.path.join(opts.checkpoint, str(i)))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, str(i), "logs"))
        vocab = build_vocab(captions, threshold=1)
        vocab_size = len(vocab)
        model_backbone = load_backbone(args)
        if args.finetune:
            if opts.resume or opts.evaluate:
                pass
            else:
                chk_filename = os.path.join(opts.pretrained, opts.selection)
                print('Loading backbone', chk_filename)
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
                model_backbone = load_pretrained_weights(model_backbone, checkpoint)
        if args.partial_train:
            model_backbone = partial_train_layers(model_backbone, args.partial_train)
        model = TextGenerator(backbone=model_backbone, embed_size=args.embed_size, hidden_size=512, vocab_size=vocab_size, num_layers=args.num_layers, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
        criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        best_acc = 0
        model_params = 0
        for parameter in model.parameters():
            model_params = model_params + parameter.numel()
        print('INFO: Trainable parameter count:', model_params)
        print('Loading dataset...')
        trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
        testloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
        # data_path = 'data/action/%s.pkl' % args.dataset

        train_alphapose_dataset = AlphaPoseAnnotDataset(train_json_paths, train_labels, train=True, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)
        test_alphapose_dataset = AlphaPoseAnnotDataset(test_json_paths, test_labels, train=False, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)

        train_loader = DataLoader(train_alphapose_dataset, **trainloader_params)
        test_loader = DataLoader(test_alphapose_dataset, **testloader_params)

        chk_filename = os.path.join(opts.checkpoint, str(i), "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            # dont load the model
            # model.load_state_dict(checkpoint['model'], strict=True)

        if not opts.evaluate:
            optimizer = optim.AdamW(
                [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                    {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
                ],      lr=args.lr_backbone,
                        weight_decay=args.weight_decay
            )

            scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
            st = 0
            print('INFO: Training on {} batches'.format(len(train_loader)))
            if opts.resume:
                st = checkpoint['epoch']
                if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
                lr = checkpoint['lr']
                if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                    best_acc = checkpoint['best_acc']
            # Training
            all_accs_train, all_loss_train, all_accs_test, all_loss_test = [], [], [], []
            for epoch in range(st, args.epochs):
                print('Training epoch %d.' % epoch)
                losses_train = AverageMeter()
                accs_train = AverageMeter()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                model.train()
                end = time.time()
                iters = len(train_loader)
                for idx, (batch_input, captions, lengths) in enumerate(train_loader):    # (N, 2, T, 17, 3)
                    data_time.update(time.time() - end)
                    batch_size = len(batch_input)
                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                    if torch.cuda.is_available():
                        targets = targets.cuda()
                        batch_input = batch_input.cuda()
                    output = model(batch_input, captions, length)
                    optimizer.zero_grad()
                    loss_train = criterion(output, targets)
                    losses_train.update(loss_train.item(), batch_size)
                    loss_train.backward()
                    optimizer.step()
                    batch_time.update(time.time() - end)
                    end = time.time()
                if (idx + 1) % opts.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Binary_Acc {accs_train.val:.3f} ({accs_train.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses_train, accs_train=accs_train))
                    sys.stdout.flush()

                test_loss, test_acc = validate(test_loader, model, criterion)

                train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
                train_writer.add_scalar('train_acc', accs_train.avg, epoch + 1)
                train_writer.add_scalar('test_loss', test_loss, epoch + 1)
                train_writer.add_scalar('test_acc', test_acc, epoch + 1)
                all_accs_train.append(accs_train.avg)
                all_loss_train.append(losses_train.avg)
                all_accs_test.append(test_acc)
                all_loss_test.append(test_loss)

                scheduler.step()

                # Save latest checkpoint.
                chk_path = os.path.join(opts.checkpoint, str(i), 'latest_epoch.bin')
                print('Saving checkpoint to', chk_path)
                torch.save({
                    'epoch': epoch+1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc' : best_acc
                }, chk_path)

                # Save best checkpoint.
                best_chk_path = os.path.join(opts.checkpoint, str(i), 'best_epoch.bin'.format(epoch))
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("save best checkpoint")
                    torch.save({
                    'epoch': epoch+1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc' : best_acc
                    }, best_chk_path)

            # display as image
            display_train_test_results(os.path.join("vis"), i, all_accs_train, all_loss_train, all_accs_test, all_loss_test)

            kcv_results[i] = {
                'train_accs': all_accs_train,
                'train_losses': all_loss_train,
                'test_accs': all_accs_test,
                'test_losses': all_loss_test
            }

    print_kcv_results(kcv_results)

    if opts.evaluate:
        test_loss, test_acc, test_top5 = validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'.format(loss=test_loss, top1=test_acc))

def train_with_config(args, opts):
    print('INFO: Training with all data')
    all_json_paths, labels = get_data(os.path.join('data', 'walking'))
    train_json_paths, train_labels, test_json_paths, test_labels = all_json_paths, labels, all_json_paths, labels
    print(args)
    try:
        os.makedirs(os.path.join(opts.checkpoint, "all"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "all", "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = WalkingNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    # data_path = 'data/action/%s.pkl' % args.dataset

    train_alphapose_dataset = AlphaPoseDataset(train_json_paths, train_labels, train=True, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)
    test_alphapose_dataset = AlphaPoseDataset(test_json_paths, test_labels, train=False, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)

    train_loader = DataLoader(train_alphapose_dataset, **trainloader_params)
    test_loader = DataLoader(test_alphapose_dataset, **testloader_params)

    chk_filename = os.path.join(opts.checkpoint, "all", "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        # dont load the model
        # model.load_state_dict(checkpoint['model'], strict=True)

    if not opts.evaluate:
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone,
                    weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        all_accs_train, all_loss_train, all_accs_test, all_loss_test = [], [], [], []
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            accs_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc_train = binary_accuracy(output, batch_gt)
                accs_train.update(acc_train, batch_size)
                loss_train.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Binary_Acc {accs_train.val:.3f} ({accs_train.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses_train, accs_train=accs_train))
                sys.stdout.flush()

            test_loss, test_acc = validate(test_loader, model, criterion)

            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_acc', accs_train.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_acc', test_acc, epoch + 1)
            all_accs_train.append(accs_train.avg)
            all_loss_train.append(losses_train.avg)
            all_accs_test.append(test_acc)
            all_loss_test.append(test_loss)

            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, "all", 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, "all", 'best_epoch.bin'.format(epoch))
            if test_acc > best_acc:
                best_acc = test_acc
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

        # display as image
        display_train_test_results(os.path.join("vis"), "all", all_accs_train, all_loss_train, all_accs_test, all_loss_test)

    if opts.evaluate:
        test_loss, test_acc, test_top5 = validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'.format(loss=test_loss, top1=test_acc))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    if opts.kcv:
        train_with_config_kcv(args, opts)
    else:
        train_with_config(args, opts)
