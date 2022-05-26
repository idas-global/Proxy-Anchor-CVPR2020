import random

import numpy as np

from .base import *
import scipy.io

from .note_families import slice_to_make_set


class Cars(BaseDataset):
    def __init__(self, root, mode, seed, le, transform = None):
        self.name = 'cars'
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform
        self.label_encoder = None

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))

        self.class_names = list([cars['class_names'][0][item[-2][0][0] - 1][0] for item in cars['annotations'][0]])
        self.class_names_coarse = [name.split(' ')[0] if name.split(' ')[0] != 'Land'
                                   else ''.join(name.split(' ')[0:2]) for name in self.class_names]
        self.class_names_fine = [name for name in [' '.join(name.split(' ')[0:-1]) for name in self.class_names]]

        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]

        chosen_idxs = self.choose_train_test_slice(seed, ys)

        im_paths = [a[0][0] for a in cars['annotations'][0]]
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)

        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))
        self.class_names_fine_dict = dict(zip(self.ys, self.class_names_fine))

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys']:
            setattr(self, param, slice_to_make_set(chosen_idxs, getattr(self, param)))

    def choose_train_test_slice(self, seed, ys):
        if self.mode == 'train' or self.mode == 'validation':
            self.classes = range(0, 98)
            observations = [i for i, y in zip(self.class_names, ys) if y in self.classes]

            random.seed(seed)
            chosen_idxs = random.choices(range(len(observations)), k=int(round(0.8 * len(observations))))

            if self.mode == 'validation':
                chosen_idxs = [i for i in range(len(observations)) if i not in chosen_idxs]

        elif self.mode == 'eval':
            self.classes = range(98, 196)
            observations = [i for i, y in zip(self.class_names, ys) if y in self.classes]
            chosen_idxs = list(range(len(observations)))
        return chosen_idxs