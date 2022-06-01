import random

from .base import *
from .note_families import slice_to_make_set


class CUBirds(BaseDataset):
    def __init__(self, root, mode, seed, le, transform=None):
        self.label_encoder = None
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.name = 'CUB'
        self.transform = transform

        if self.mode == 'train':
            self.classes = range(0, 100)
            self.perplex = 30
        elif self.mode == 'validation':
            self.classes = range(0, 100)
            self.perplex = 20
        elif self.mode == 'eval':
            self.classes = range(100, 200)
            self.perplex = 55

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0

        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: the full path to an image
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(i[0])
                index += 1

        self.class_names = self.im_paths
        self.class_names_coarse = [parse_im_name(specific_species, exclude_trailing_consonants=False)
                                   for specific_species in self.class_names]
        self.class_names_fine = [parse_im_name(specific_species, exclude_trailing_consonants=False, fine=True)
                                   for specific_species in self.class_names]

        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))
        self.class_names_fine_dict = dict(zip(self.ys, self.class_names_fine))
        self.tsne_labels = ['_'.join(os.path.split(i)[-1].split('_')[0:4]) for i in self.im_paths]

        chosen_idxs = self.choose_train_test_slice(seed, self.ys)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys', 'tsne_labels']:
            setattr(self, param, slice_to_make_set(chosen_idxs, getattr(self, param)))

    def choose_train_test_slice(self, seed, ys):
        if self.mode == 'train' or self.mode == 'validation':
            observations = [i for i, y in zip(self.class_names, ys) if y in self.classes]
            random.seed(seed)
            chosen_idxs = random.choices(range(len(observations)), k=int(round(0.8 * len(observations))))

            if self.mode == 'validation':
                chosen_idxs = [i for i in range(len(observations)) if i not in chosen_idxs]

        elif self.mode == 'eval':
            observations = [i for i, y in zip(self.class_names, ys) if y in self.classes]
            chosen_idxs = list(range(len(observations)))
        return chosen_idxs

def parse_im_name(specific_species, exclude_trailing_consonants=False, fine=False):
    if fine:
        filter = os.path.split(os.path.split(specific_species)[0])[1].split('.')[-1].lower()
    else:
        coarse_filter = os.path.split(os.path.split(specific_species)[0])[1].split('_')[-1].lower()
        if '.' in coarse_filter:
            coarse_filter = coarse_filter.split('.')[-1]
        filter = coarse_filter

    if exclude_trailing_consonants:
        if filter[-1].isalpha():
            filter = filter[0:-1]
    return filter
