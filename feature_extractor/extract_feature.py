# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import os
import time

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from src.data.loader import load_data, get_data_transformations, KFold, per_target
from src.model.model_factory import model_factory, to_cuda, sgd_optimizer
from src.model.pretrain import load_pretrained
from src.data.VOC2007 import VOC2007_dataset
from src.utils import (bool_flag, init_distributed_mode, initialize_exp, AverageMeter,
                       restart_from_checkpoint, fix_random_seeds,)

logger = getLogger()


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train a linear classifier on conv layer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Experiment dump path")
    parser.add_argument('--epoch', type=int, default=0,
                        help='Current epoch to run')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='First iter to run in the current epoch')

    # model params
    parser.add_argument('--pretrained', type=str, default='',
                        help='Use this instead of random weights.')
    parser.add_argument('--conv', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        help='On top of which layer train classifier.')

    # datasets params
    parser.add_argument('--data_path', type=str, default='',
                        help='Where to find supervised dataset')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of data loading workers')
    parser.add_argument('--sobel', type=bool_flag, default=False)

    # optim params
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Max number of epochs to run')
    parser.add_argument('--batch_size', default=64, type=int)

    # model selection
    parser.add_argument('--split', type=str, required=False, default='trainval', choices=['train', 'trainval'],
                        help='for PASCAL dataset, train on train or train+val')
    parser.add_argument('--kfold', type=int, default=None,
                        help="""dataset randomly partitioned into kfold equal sized subsamples.
                        Default None: no cross validation: train on full train set""")
    parser.add_argument('--cross_valid', type=int, default=None,
                        help='between 0 and kfold - 1: index of the round of cross validation')

    # distributed training params
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug within a SLURM job")

    return parser.parse_args()


def main(args):

    # initialize the experiment
    logger, training_stats = initialize_exp(args, 'epoch', 'iter', 'prec',
                                            'loss', 'prec_val', 'loss_val')

    if not 'VOC2007' in args.data_path:
        main_data_path = args.data_path
        args.data_path = os.path.join(main_data_path, 'train')
        train_dataset = load_data(args)
    else:
        train_dataset = VOC2007_dataset(args.data_path, split=args.split)

    args.test = 'val' if args.split == 'train' else 'test'
    if not 'VOC2007' in args.data_path:
        if args.cross_valid is None:
            args.data_path = os.path.join(main_data_path, 'val')
        val_dataset = load_data(args)
    else:
        val_dataset = VOC2007_dataset(args.data_path, split=args.test)

    if args.cross_valid is not None:
        kfold = KFold(per_target(train_dataset.imgs), args.cross_valid, args.kfold)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=kfold.train,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=kfold.val,
            num_workers=args.workers)

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

    # prepare the different data transformations
    tr_val, tr_train = get_data_transformations()

    #data preprocess transformation could keep in consistency
    train_dataset.transform = tr_val
    val_dataset.transform = tr_val

    # build model skeleton
    fix_random_seeds()
    model = model_factory(args)

    # keep only conv layers
    model.body.classifier = None

    load_pretrained(model, args)

    print('feature at conv{} is extracting!'.format(args.conv))
    model.conv = args.conv

    if 'places' in args.data_path:
        nmb_classes = 205
    elif 'VOC2007' in args.data_path:
        nmb_classes = 20
    else:
        nmb_classes = 1000

    # distributed training wrappere)
    model = model.cuda()

    logger.info('model to cuda')

    # set optimizer
    optimizer = sgd_optimizer(reglog, args.lr, args.wd)

    ## variables to reload to fetch in checkpoint
    to_restore = {'epoch': 0, 'start_iter': 0}


    model.eval()

    save_feature(args, model, optimizer, train_loader, 'trainval')
    save_feature(args, model, optimizer, val_loader, 'test')

    print('save finished')
   

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, num_labels, conv):
        super(RegLog, self).__init__()
        if conv < 3:
            av = 18
            s = 9216
        elif conv < 5:
            av = 14
            s = 8192
        elif conv < 8:
            av = 9
            s = 9216
        elif conv < 11:
            av = 6
            s = 8192
        elif conv < 14:
            av = 3
            s = 8192
        self.av_pool = nn.AvgPool2d(av, stride=av, padding=0)
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def save_feature(args, model, reglog, optimizer, loader, split):
    feature_output = []
    label_output = []

    print(len(loader))
    for iter_epoch, (inp, target) in enumerate(loader):
        inp = inp.cuda(non_blocking=True)

        output = model(inp)

        output = F.adaptive_max_pool2d(output, (4,4))

        output = output.cpu().detach().numpy().tolist()
        target = target.cpu().detach().numpy().tolist()

        for i in range(len(output)):
            feature_output.append(output[i])
            label_output.append(target[i])
    feature_output = np.array(feature_output)
    label_output = np.array(label_output)
    print(feature_output.shape, label_output.shape)
    feature_output = feature_output.reshape(feature_output.shape[0],-1)
    label_output = label_output.reshape(feature_output.shape[0],-1)

    if not os.path.exists(os.path.join(args.dump_path, 'train')):
        os.mkdir(os.path.join(args.dump_path, 'train'))
    if not os.path.exists(os.path.join(args.dump_path, 'test')):
        os.mkdir(os.path.join(args.dump_path, 'test'))

    if split == 'trainval':
        out_feat_file = os.path.join(args.dump_path, 'train', 'features.npy')
        out_target_file = os.path.join(args.dump_path, 'train', 'targets.npy')
    elif split == 'test':
        out_feat_file = os.path.join(args.dump_path, 'test', 'features.npy')
        out_target_file = os.path.join(args.dump_path, 'test', 'targets.npy')


    print('feature shape(after reshape) {} saved at {}'.format(feature_output.shape, out_feat_file))

    np.save(out_feat_file, feature_output)
    np.save(out_target_file, label_output)


if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()

    # run experiment
    main(args)
