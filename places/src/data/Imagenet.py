import os
import zipfile

import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data

def loader(path, file_img):
    img = Image.open(os.path.join(path,file_img))
    return img.convert('RGB')

def load_data(file_name):
  file = open(file_name, 'r')
  line = file.readline()
  datasets = []

  while line:
    datasets.append(line.strip().split(' '))
    line = file.readline()
  datasets = np.array(datasets)
  return datasets


class Imagenet_dataset(data.Dataset):

    def __init__(self, root,size, transform=None):
        self.root = root
        self.transform = transform
        self.sub_classes = None

        self.indexes = load_data(os.path.join(self.root, 'meta/train.txt'))
        self.indexes = self.indexes[:size,0]

        # for subsets
        self.subset_indexes = None
        self.datapath = self.root + '/train'
        
    
    def __getitem__(self, ind):
        index = ind
        if self.subset_indexes is not None:
            index = self.subset_indexes[ind]
        name = self.indexes[index]
        img = loader(self.datapath,name)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # id of cluster
        sub_class = -100
        if self.sub_classes is not None:
            sub_class = self.sub_classes[ind]

        return img, sub_class

    def __len__(self):
        if self.subset_indexes is not None:
            return len(self.subset_indexes)
        return len(self.indexes)
                                    



