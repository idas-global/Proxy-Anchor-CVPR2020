import os
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow
import albumentations as alb
import numpy as np
import scipy.io


def transform(dataset, image, train):
    sz_resize = 256
    sz_crop = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if dataset.is_inception:
        sz_resize = 256
        sz_crop = 224
        mean = (104, 117, 128)
        std = (1, 1, 1)

    p = 1
    if dataset.name == 'NoteStyles':
        p = 0

    transformed = transform_image(image, p, sz_crop, sz_resize, train)

    for i in range(len(transformed.shape)):
        transformed[:, :, i] = (transformed[:, :, i] - mean[i]) / std[i]

    return transformed


def transform_image(image, p, sz_crop, sz_resize, train=True):
    if train:
        transform = alb.Compose([
            alb.RandomResizedCrop(sz_crop, sz_crop, scale=(0.7, 1), always_apply=True),
            alb.HorizontalFlip(p=0.5),
            alb.transforms.Resize(sz_resize, sz_resize),
            alb.GaussNoise(p=0.1),
            alb.GaussianBlur(p=0.1),
            alb.RandomBrightnessContrast(p=0.1),
            alb.RandomShadow(p=0.1),
            alb.RandomRain(p=0.1),
            alb.GridDistortion(p=0.1),

            #alb.VerticalFlip(p=p / 2),
            alb.CenterCrop(sz_crop, sz_crop, p=p),


        ], p=1)
    else:
        transform = alb.Compose([
            alb.transforms.Resize(sz_resize, sz_resize),
            alb.CenterCrop(sz_crop, sz_crop, p=p),
        ], p=1)
    transformed = transform(image=image)['image'].astype('float32')
    return transformed


class NoteStyles(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, args, seed, shuffle=True, mode='train', is_inception=True):
        self.root = "D:/Rupert_Book_Augmented/"
        'Initialization'
        self.name = 'NoteStyles'
        self.mode = mode
        self.batch_size = args.sz_batch
        self.shuffle = shuffle
        self.sz_embedding = args.sz_embedding
        self.is_inception = is_inception
        self.im_dimensions = (224, 224, 3) # TODO Put in parser
        self.im_paths = []

        for (root, dirs, files) in os.walk(self.root):
            for file in files:
                if '.bmp' in file or '.jpg' in file or '.png' in file:
                    self.im_paths.append(os.path.join(root, file))

        self.class_names = [os.path.split(os.path.split(item)[0])[-1] for item in self.im_paths]
        self.class_names_coarse = [name.split('_')[0] for name in self.class_names]
        self.class_names_fine = [i[0:-1] if i[-1].isalpha() else i for i in self.class_names]

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.ys = le.fit_transform(self.class_names_fine)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        self.nb_classes = len(le_name_mapping.keys())

        if self.mode == 'train' or self.mode == 'val':
            self.classes = [label for key, label in le_name_mapping.items() if not key.startswith('100_')]
        else:
            self.classes = [label for key, label in le_name_mapping.items()]

        chosen_images = [idx for idx, i in enumerate(self.ys) if i in self.classes]
        random.seed(seed)
        if self.mode == 'train':
            chosen_images = random.choices(chosen_images, k=int(np.round(0.8*len(chosen_images))))
        if self.mode == 'val':
            not_chosen_images = random.choices(chosen_images, k=int(np.round(0.8*len(chosen_images))))
            chosen_images = [i for i in chosen_images if i not in not_chosen_images]

        self.dataset_size = len(chosen_images)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys']:
            setattr(self, param, self.slice_to_make_set(chosen_images, getattr(self, param)))

        if shuffle:
            temp = list(zip(self.im_paths, self.class_names, self.class_names_coarse, self.class_names_fine, self.ys))
            random.shuffle(temp)
            self.im_paths, self.class_names, self.class_names_coarse, self.class_names_fine, self.ys = zip(*temp)

    def slice_to_make_set(self, chosen_images, param):
        return list(np.array(param)[chosen_images])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_slice = range(len(self.im_paths))[index * self.batch_size : (index + 1) * self.batch_size]

        imgs_for_batch = self.slice_to_make_set(batch_slice, self.im_paths)
        y = self.slice_to_make_set(batch_slice, self.ys)

        x = np.empty((len(imgs_for_batch), *self.im_dimensions))

        for idx, i in enumerate(imgs_for_batch):
            image = cv2.imread(i)
            x[idx] = transform(self, image)

        #y = to_categorical(np.array(y).astype(np.float32).reshape(-1, 1), num_classes=self.nb_classes)
        return [x, np.array(y)]


class Cars(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, args, seed, shuffle=True, mode='train', is_inception=True, le=None):
        os.chdir('../data/')
        data_root = os.getcwd()
        self.shape = args.sz_batch
        self.root = data_root + '/cars196'
        self.name = 'Cars'
        self.mode = mode
        self.batch_size = args.sz_batch
        self.shuffle = shuffle
        self.sz_embedding = args.sz_embedding
        self.is_inception = is_inception
        self.im_dimensions = (224, 224, 3) # TODO Put in parser
        self.im_paths = []

        for (root, dirs, files) in os.walk(self.root):
            for file in files:
                if '.bmp' in file or '.jpg' in file or '.png' in file:
                    self.im_paths.append(os.path.join(root, file))

        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        self.class_names = list([cars['class_names'][0][item[-2][0][0] - 1][0] for item in cars['annotations'][0]])
        self.class_names_coarse = [name.split(' ')[0] if name.split(' ')[0] != 'Land'
                                   else ''.join(name.split(' ')[0:2]) for name in self.class_names]
        self.class_names_fine = [name for name in [' '.join(name.split(' ')[0:-1]) for name in self.class_names]]

        from sklearn import preprocessing
        if le is None:
            le = preprocessing.LabelEncoder()
        le.fit(self.class_names_fine)
        self.ys = le.transform(self.class_names_fine)
        self.label_encoder = le

        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        self.nb_classes_total = len(le_name_mapping.keys())

        if self.mode == 'train' or self.mode == 'val':
            self.classes = range(0, int(np.round(self.nb_classes_total / 2)))
        else:
            self.classes = range(int(np.round(self.nb_classes_total / 2)), self.nb_classes_total - 1)

        chosen_images = [idx for idx, i in enumerate(self.ys) if i in self.classes]
        random.seed(seed)
        if self.mode == 'train':
            chosen_images = random.choices(chosen_images, k=int(np.round(0.8*len(chosen_images))))
        if self.mode == 'val':
            not_chosen_images = random.choices(chosen_images, k=int(np.round(0.8*len(chosen_images))))
            chosen_images = [i for i in chosen_images if i not in not_chosen_images]

        self.dataset_size = len(chosen_images)

        for param in ['im_paths', 'class_names', 'class_names_coarse', 'class_names_fine', 'ys']:
            setattr(self, param, self.slice_to_make_set(chosen_images, getattr(self, param)))
 
        if shuffle:
            temp = list(zip(self.im_paths, self.class_names, self.class_names_coarse, self.class_names_fine, self.ys))
            random.shuffle(temp)
            self.im_paths, self.class_names, self.class_names_coarse, self.class_names_fine, self.ys = zip(*temp)

        self.nb_classes = len(np.unique(self.ys, axis=0))

    def slice_to_make_set(self, chosen_images, param):
        return list(np.array(param)[chosen_images])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_slice = range(len(self.im_paths))[index * self.batch_size : (index + 1) * self.batch_size]

        imgs_for_batch = self.slice_to_make_set(batch_slice, self.im_paths)
        y = self.slice_to_make_set(batch_slice, self.ys)

        x = np.empty((len(imgs_for_batch), *self.im_dimensions))

        for idx, i in enumerate(imgs_for_batch):
            image = cv2.imread(i)
            #x[idx] = transform(self, image, self.mode == 'train')
            x[idx] = transform(self, image, True)
        #y = to_categorical(np.array(y).astype(np.float32).reshape(-1, 1), num_classes=self.nb_classes)
        return [x, np.array(y)]