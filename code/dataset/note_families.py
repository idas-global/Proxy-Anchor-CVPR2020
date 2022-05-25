import random

from sklearn.preprocessing import LabelEncoder

from .base import *

def slice_to_make_set(chosen_images, param):
    return list(np.array(param)[chosen_images])


class Families(BaseDataset):
    def __init__(self, root, mode, seed, le, transform = None, plate='front'):
        self.name = 'note_families'

        if mode == 'train':
            self.root = f'/mnt/ssd1/Genesys_2_Capture/1604_{plate}s_augmented/'
            self.root = f'D:/1604_{plate}s_augmented/'
        if mode == 'validation':
            #self.root = 'D:/Rupert_Book_Augmented/'
            self.root = f'/mnt/ssd1/Genesys_2_Capture/1604_{plate}s_augmented/'
            self.root = f'D:/1604_{plate}s_augmented/'

        self.mode = mode
        self.transform = transform
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

        if self.mode == 'train' or self.mode == 'validation':
            random.seed(seed)
            chosen_idxs = random.choices(range(len(self.class_names)), k=int(round(0.8*len(self.class_names))))

            if self.mode == 'validation':
                chosen_idxs = [i for i in range(len(self.class_names)) if i not in chosen_idxs]

        self.ys = le.transform(self.class_names_fine)
        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))
        self.classes = set(self.ys)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys']:
            setattr(self, param, slice_to_make_set(chosen_idxs, getattr(self, param)))

