from .base import *


class Notes(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = "D:/Rupert_Book_Augmented"
        self.mode = mode
        self.name = 'NoteStyles'
        self.transform = transform

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        for i in torchvision.datasets.ImageFolder(root=self.root).imgs:
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if fn[:2] != '._':
                self.im_paths.append(i[0])

        self.class_names = [os.path.split(os.path.split(item)[0])[-1] for item in self.im_paths]
        self.class_names_coarse = [name.split('_')[0] for name in self.class_names]
        self.class_names_fine = [i[0:-1] if i[-1].isalpha() else i for i in self.class_names]
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.ys = le.fit(self.class_names_fine)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        if self.mode == 'train':
            self.classes = [label for key, label in le_name_mapping.items() if not key.startswith('100_')]
        if self.mode == 'val':
            self.classes = [label for key, label in le_name_mapping.items() if not key.startswith('100_')]
        elif self.mode == 'eval':
            # TODO Stop fitting to train
            self.classes = [label for key, label in le_name_mapping.items() if key.startswith('100_')]

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        for i in torchvision.datasets.ImageFolder(root=self.root).imgs:
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if fn[:2] != '._':
                self.im_paths.append(i[0])

        self.class_names = [os.path.split(os.path.split(item)[0])[-1] for item in self.im_paths]
        self.class_names_coarse = [name.split('_')[0] for name in self.class_names]
        self.class_names_fine = [i[0:-1] if i[-1].isalpha() else i for i in self.class_names]
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.ys = le.fit(self.class_names_fine)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        if self.mode == 'train':
            self.classes = [label for key, label in le_name_mapping.items() if not key.startswith('100_')]
        if self.mode == 'val':
            self.classes = [label for key, label in le_name_mapping.items() if not key.startswith('100_')]
        elif self.mode == 'eval':
            # TODO Stop fitting to train
            self.classes = [label for key, label in le_name_mapping.items() if key.startswith('100_')]