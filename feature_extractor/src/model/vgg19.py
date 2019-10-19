# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn
import torch.nn.init as init

#import sys
#path_torchE = "/mnt/lustre/share/miniconda3/envs/r0.1.2/lib/python3.6/site-packages/torchE"
#sys.path.insert(0, path_torchE)
#from nn import SyncBatchNorm2d

#BatchNorm = None


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG19(nn.Module):
    '''
    VGG16 model 
    '''
    def __init__(self, dim_in, relu=True, dropout=0.5, batch_norm=True, syncbn=False, group_size=1, group=None, sync_stats=False):
        super(VGG19, self).__init__()
  
        #if syncbn:
            # from lib.syncbn import SynchronizedBatchNorm2d as BatchNorm
        #    def BNFunc(*args, **kwargs):
        #        return SyncBatchNorm2d(*args, **kwargs, eps=1e-4, momentum=0.9,
        #                               group_size=group_size, group=group,
        #                               sync_stats=sync_stats)
        #    BatchNorm = BNFunc
        #else:
        #from torch.nn import BatchNorm2d as BatchNorm

        self.features = make_layers(cfg['D'], dim_in, batch_norm=batch_norm)
        self.dim_output_space = 4096
        classifier = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
        ]
        if relu:
            classifier.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*classifier)
            
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        print('VGG19 features size after conv: {}'.format(x.size()))
        if self.classifier is not None:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
