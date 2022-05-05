from .base import *


class Notes(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = "D:/Rupert_Book_Augmented"
        self.mode = mode
        self.name = 'NoteStyles'
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0, 57)
        elif self.mode == 'eval':
            # TODO Stop fitting to train
            self.classes = range(0, 57)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=self.root).imgs:
            # i[1]: label, i[0]: the full path to an image
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(i[0])
                index += 1

        self.class_names = [os.path.split(os.path.split(item)[0])[-1] for item in self.im_paths]
        self.class_names_coarse = [name.split('_')[0] for name in self.class_names]
        self.class_names_fine = [i[0:-1] if i[-1].isalpha() else i for i in self.class_names]