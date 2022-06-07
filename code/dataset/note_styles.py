import os
import random
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .base import *
import scipy.io

class Notes(BaseDataset):
    def __init__(self, root, mode, seed, le, transform = None):
        self.name = 'note_styles'
        self.mode = mode
        self.transform = transform

        if sys.platform == 'linux':
            if mode == 'train':
                self.root = '/mnt/ssd1/Genesys_2_Capture/rupert_book_augmented/'
                self.perplex = 30
            if mode == 'validation':
                self.root = '/mnt/ssd1/Genesys_2_Capture/rupert_book_augmented_test/'
                self.perplex = 10
        else:
            if mode == 'train':
                self.root = 'D:/raw_data/rupert_book/rupert_book_augmented/'
                self.perplex = 30
            if mode == 'validation':
                self.root = 'D:/raw_data/rupert_book/rupert_book_augmented_test/'
                self.perplex = 10

        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        im_paths = []
        for (root, dirs, files) in os.walk(self.root):
            for file in files:
                if '.bmp' in file:
                    im_paths.append(os.path.join(root, file))

        self.class_names = [os.path.split(os.path.split(i)[0])[-1] for i in im_paths]
        self.class_names_coarse = [name.split('_')[0] for name in self.class_names]
        self.class_names_fine = [name[0:-1] if name[-1].isalpha() else name for name in self.class_names]

        if le is None:
            le = LabelEncoder()
            le.fit(self.class_names_fine)
        self.label_encoder = le

        self.ys = le.transform(self.class_names_fine)
        self.class_names_fine_dict = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        self.class_names_coarse_dict = dict(zip(range(len(self.label_encoder.classes_)), [name.split('_')[0] for name in self.label_encoder.classes_]))
        self.class_names_coarse_dict[9999] = 'oversaturated'
        self.class_names_fine_dict[9999] = 'oversaturated'

        self.im_paths = im_paths
        self.classes = set(self.ys)
        self.tsne_labels = ['_'.join(os.path.split(i)[-1].split('_')[0:4]) for i in self.im_paths]
