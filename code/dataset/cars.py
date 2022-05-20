import random

from .base import *
import scipy.io
from sklearn import preprocessing


class Cars(BaseDataset):
    def __init__(self, root, mode, args, seed, le, transform = None):
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        os.chdir('../data/')
        data_root = os.getcwd()
        self.shape = args.sz_batch
        self.root = data_root + '/cars196'
        self.name = 'Cars'
        self.mode = mode
        self.batch_size = args.sz_batch
        self.sz_embedding = args.sz_embedding
        self.im_dimensions = (3, 224, 224)  # TODO Put in parser
        self.im_paths = []

        for (root, dirs, files) in os.walk(self.root):
            for file in files:
                if '.bmp' in file or '.jpg' in file or '.png' in file:
                    self.im_paths.append(os.path.join(root, file))

        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        self.class_names = list([cars['class_names'][0][item[-2][0][0] - 1][0] for item in cars['annotations'][0]])

        self.class_names_fine = [name for name in [' '.join(name.split(' ')[0:-1]) for name in self.class_names]]
        self.class_names_coarse = [name.split(' ')[0] if name.split(' ')[0] != 'Land'
                                   else ''.join(name.split(' ')[0:2]) for name in self.class_names]

        self.ys, le = self.create_labels(le, self.class_names_fine)
        self.label_encoder = le

        self.class_names_coarse_dict = dict(zip(self.ys, self.class_names_coarse))

        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        self.nb_classes_total = len(le_name_mapping.keys())

        if self.mode == 'train' or self.mode == 'val':
            self.classes = range(0, int(np.round(self.nb_classes_total / 2)))
        else:
            self.classes = range(int(np.round(self.nb_classes_total / 2)), self.nb_classes_total - 1)

        chosen_images = [idx for idx, i in enumerate(self.ys) if i in self.classes]
        random.seed(seed)
        if self.mode == 'train':
            chosen_images = random.choices(chosen_images, k=int(np.round(0.8 * len(chosen_images))))
            chosen_images = np.sort(chosen_images)
        if self.mode == 'val':
            not_chosen_images = random.choices(chosen_images, k=int(np.round(0.8 * len(chosen_images))))
            chosen_images = [i for i in chosen_images if i not in not_chosen_images]
            chosen_images = np.sort(chosen_images)

        self.dataset_size = len(chosen_images)

        random.shuffle(chosen_images)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys']:
            setattr(self, param, self.slice_to_make_set(chosen_images, getattr(self, param)))

        self.nb_classes = len(np.unique(self.ys, axis=0))

    def create_labels(self, le, labels):
        if le is None:
            le = preprocessing.LabelEncoder()
            le.fit(labels)
        return le.transform(labels), le


    def slice_to_make_set(self, chosen_images, param):
        return list(np.array(param)[chosen_images])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_slice = range(len(self.im_paths))[index * self.batch_size: (index + 1) * self.batch_size]

        imgs_for_batch = self.slice_to_make_set(batch_slice, self.im_paths)
        y = self.slice_to_make_set(batch_slice, self.ys)

        x = np.empty((len(imgs_for_batch), *self.im_dimensions))

        for idx, i in enumerate(imgs_for_batch):
            im = PIL.Image.open(i)
            # convert gray to rgb
            if len(list(im.split())) == 1:
                im = im.convert('RGB')

            if self.transform is not None:
                x[idx, :] = self.transform(im)
        # y = to_categorical(np.array(y).astype(np.float32).reshape(-1, 1), num_classes=self.nb_classes)
        return x, np.array(y)
