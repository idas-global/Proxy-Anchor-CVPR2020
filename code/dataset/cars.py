from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/cars196'
        self.mode = mode
        self.name = 'CARS'
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)
                
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        self.class_names = list([cars['class_names'][0][item[-2][0][0] - 1][0] for item in cars['annotations'][0]])
        self.class_names_coarse = [name.split(' ')[0] if name.split(' ')[0] != 'Land'
                                   else ''.join(name.split(' ')[0:2]) for name in self.class_names]
        self.class_names_fine = [name for name in [' '.join(name.split(' ')[0:-1]) for name in self.class_names]]

        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1