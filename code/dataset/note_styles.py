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

        if sys.platform == 'linux':
            if mode == 'train':
                self.root = '/mnt/ssd1/Rupert_Book_Augmented/'
            if mode == 'validation':
                self.root = '/mnt/ssd1/Rupert_Book_Augmented_Test/'
        else:
            if mode == 'train':
                self.root = 'D:/Rupert_Book_Augmented/'
            if mode == 'validation':
                self.root = 'D:/Rupert_Book_Augmented/'

        self.mode = mode
        self.transform = transform
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
        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))
        self.class_names_fine_dict = dict(zip(self.ys, self.class_names_fine))

        self.im_paths = im_paths
        self.classes = set(self.ys)

