from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'imagesPlaces205_resize')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(int(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)



