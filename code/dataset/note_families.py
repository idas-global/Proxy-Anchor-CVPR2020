import random
import sys

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from .base import *

def slice_to_make_set(chosen_images, param):
    return list(np.array(param)[chosen_images])


class Families(BaseDataset):
    def __init__(self, root, mode, seed, le, transform = None, plate='front'):
        self.name = 'note_families'
        self.mode = mode
        self.transform = transform
        if sys.platform == 'linux':
            if mode == 'train':
                self.root = f'/mnt/ssd1/Genesys_2_Capture/1604_{plate}s_augmented/'
                self.perplex = 50
            if mode == 'validation':
                self.root = f'/mnt/ssd1/Genesys_2_Capture/1604_{plate}s_augmented/'
                self.perplex = 20
            if mode == 'eval':
                self.root = f'/mnt/ssd1/Genesys_2_Capture/1604_{plate}s_augmented/'
                self.perplex = 50
        else:
            if mode == 'train':
                self.root = f'D:/1604_{plate}s_augmented/'
                self.perplex = 50
            if mode == 'validation':
                self.root = f'D:/1604_{plate}s_augmented/'
                self.perplex = 20
            if mode == 'eval':
                self.root = f'D:/1604_{plate}s_augmented/'
                self.perplex = 50

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        im_paths = []
        for (root, dirs, files) in os.walk(self.root):
            for file in files:
                if '.bmp' in file:
                    im_paths.append(os.path.join(root, file))
        self.im_paths = im_paths

        self.class_names = [os.path.split(os.path.split(item)[0])[-1] for item in self.im_paths]
        self.class_names_coarse = [i[0:-1] if i[-1].isalpha() and 'GENUINE' not in i else i
                                   for i in [name.split('_')[0] for name in self.class_names]]
        self.class_names_fine = [i[0:-1] if i[-1].isalpha() and 'GENUINE' not in i else i for i in self.class_names]

        if le is None:
            le = LabelEncoder()
            le.fit(self.class_names_fine)
        self.label_encoder = le

        self.ys = le.transform(self.class_names_fine)
        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))
        self.class_names_fine_dict = dict(zip(self.ys, self.class_names_fine))

        self.tsne_labels = ['_'.join(os.path.split(i)[-1].split('_')[0:4]) for i in self.im_paths]

        chosen_idxs = self.choose_train_test_slice(seed)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys', 'tsne_labels']:
            setattr(self, param, slice_to_make_set(chosen_idxs, getattr(self, param)))

        self.classes = set(self.ys)

    def choose_train_test_slice(self, seed):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for train_index, test_index in sss.split(range(len(self.class_names)), self.class_names_fine):
            if self.mode == 'train':
                chosen_idxs = train_index
            if self.mode == 'validation':
                chosen_idxs = test_index
            if self.mode == 'eval':
                chosen_idxs = np.hstack((train_index, test_index))
        return chosen_idxs
