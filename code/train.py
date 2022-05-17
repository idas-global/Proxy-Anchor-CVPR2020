import argparse, os
import random, dataset, utils, losses

import cv2
import scipy
import tensorflow
from tqdm import tqdm

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
import torch
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
import argparse, os
import urllib

import utils, losses
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from generator import NoteStyles, Cars

import tensorflow_addons as tfa
import tensorflow_hub as hub
import albumentations as alb


def configure_parser():
    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                     + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
                                     )
    # export directory, training and val datasets, test datasets
    parser.add_argument('--LOG_DIR',
                        default='../logs',
                        help='Path to log folder'
                        )
    parser.add_argument('--dataset',
                        default='cub',
                        help='Training dataset, e.g. cub, cars, SOP, Inshop'
                        )
    parser.add_argument('--embedding-size', default=512, type=int,
                        dest='sz_embedding',
                        help='Size of embedding that is appended to backbone model.'
                        )
    parser.add_argument('--batch-size', default=150, type=int,
                        dest='sz_batch',
                        help='Number of samples per batch.'
                        )
    parser.add_argument('--epochs', default=60, type=int,
                        dest='nb_epochs',
                        help='Number of training epochs.'
                        )
    parser.add_argument('--gpu-id', default=0, type=int,
                        help='ID of GPU that is used for training.'
                        )
    parser.add_argument('--workers', default=0, type=int,
                        dest='nb_workers',
                        help='Number of workers for dataloader.'
                        )
    parser.add_argument('--model', default='bn_inception',
                        help='Model for training'
                        )
    parser.add_argument('--loss', default='Proxy_Anchor',
                        help='Criterion for training'
                        )
    parser.add_argument('--optimizer', default='adamw',
                        help='Optimizer setting'
                        )
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate setting'
                        )
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='Weight decay setting'
                        )
    parser.add_argument('--lr-decay-step', default=10, type=int,
                        help='Learning decay step setting'
                        )
    parser.add_argument('--lr-decay-gamma', default=0.5, type=float,
                        help='Learning decay gamma setting'
                        )
    parser.add_argument('--alpha', default=32, type=float,
                        help='Scaling Parameter setting'
                        )
    parser.add_argument('--mrg', default=0.1, type=float,
                        help='Margin parameter setting'
                        )
    parser.add_argument('--IPC', type=int,
                        help='Balanced sampling, images per class'
                        )
    parser.add_argument('--warm', default=1, type=int,
                        help='Warmup training epochs'
                        )
    parser.add_argument('--bn-freeze', default=1, type=int,
                        help='Batch normalization parameter freeze'
                        )
    parser.add_argument('--l2-norm', default=1, type=int,
                        help='L2 normlization'
                        )
    parser.add_argument('--remark', default='',
                        help='Any reamrk'
                        )
    return parser.parse_args()


args = configure_parser()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

#
# if args.gpu_id != -1:
#     torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model,
                                                                                             args.loss,
                                                                                             args.sz_embedding,
                                                                                             args.alpha,
                                                                                             args.mrg, args.optimizer,
                                                                                             args.lr, args.sz_batch,
                                                                                             args.remark)

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset == 'note_styles':
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception'),
            crop=False
        ))
elif args.dataset != 'Inshop':
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))
else:
    trn_dataset = Inshop_Dataset(
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers=args.nb_workers,
        pin_memory=True,
        batch_sampler=batch_sampler
    )
    print('Balanced Sampling')

else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )


def create_and_compile_model(train_gen, args):
    # model = model
    y_input = Input(shape=(1,))
    backbone = tf.keras.Sequential(hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v2/classification/5", trainable=True))

                              # arguments=dict(return_endpoints=True)))
    backbone.build([None, *train_gen.im_dimensions])
    flat = tf.keras.layers.Flatten()(backbone.output)
    embed = tf.keras.layers.Dense(args.sz_embedding, kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=False, activation=None)(flat)

    criterion = losses.TF_proxy_anchor(len(set(train_gen.ys)), args.sz_embedding)
    crit_tensor = criterion([y_input, embed])

    model = Model(inputs=[backbone.input, y_input], outputs=crit_tensor)
    optimizers = [
        tfa.optimizers.AdamW(learning_rate=float(args.lr), weight_decay=args.weight_decay),
        tfa.optimizers.AdamW(learning_rate=float(args.lr)*100, weight_decay=args.weight_decay)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[0:-2]), (optimizers[1], model.layers[-2])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(optimizer=optimizer,
                  run_eagerly=False)
    return model, criterion


def create_save_dir(args):
    checkpoint_filepath = args.LOG_DIR \
                          + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset,
                                                                                                args.model,
                                                                                                args.loss,
                                                                                                args.sz_embedding,
                                                                                                args.alpha,
                                                                                                args.mrg,
                                                                                                args.optimizer,
                                                                                                args.lr, args.sz_batch,
                                                                                                args.remark)
    return checkpoint_filepath


def test_predictions(args, epoch, model, train_gen, val_gen, test_gen):
    predict_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # print('###################################')
    # print(f'######  TEST EPOCh {epoch}  #######')
    # Recalls = utils.evaluate_cos(predict_model, test_gen, epoch, args)

    print('###################################')
    print(f'###### TRAIN EPOCh {epoch}  #######')
    Recalls = utils.evaluate_cos(predict_model, train_gen, epoch, args)

    print('####################################')
    print(f'######   VAL EPOCh {epoch}  #######')
    Recalls = utils.evaluate_cos(predict_model, val_gen, epoch, args)


def prepare_layers(args, epoch, model):
    bn_freeze = args.bn_freeze
    if bn_freeze:
        for layer in model.layers:
            if layer.name == 'batch_normalization':
                layer.trainable = False
    if args.warm > 0:
        if epoch == 0:
            model.layers[-1].trainable = False
        if epoch == args.warm:
            model.layers[-1].trainable = True


def custom_loss(self, target, embeddings):
    oh_target = tf.squeeze(tf.one_hot(tf.cast(target, tf.int32), depth=self.nb_classes))
    embeddings_l2 = tf.cast(tf.nn.l2_normalize(embeddings, axis=1), tf.float32)
    proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

    pos_target = oh_target
    neg_target = 1.0 - pos_target

    pos_target = tf.cast(pos_target, tf.bool)
    neg_target = tf.cast(neg_target, tf.bool)

    cos = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

    pos_mat = tf.where(pos_target, x=tf.exp(-32 * (cos - 0.1)), y=tf.zeros_like(pos_target, dtype=tf.float32))
    neg_mat = tf.where(neg_target, x=tf.exp(32 * (cos + 0.1)), y=tf.zeros_like(neg_target, dtype=tf.float32))

    n_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(oh_target, axis=0), dtype=tf.dtypes.float32)

    pos_term = tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(pos_mat, axis=0))) / n_valid_proxies
    neg_term = tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(neg_mat, axis=0))) / tf.cast(self.nb_classes, tf.float32)
    loss = pos_term + neg_term
    return loss


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
        self.im_dimensions = (224, 224, 3)  # TODO Put in parser
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
            chosen_images = random.choices(chosen_images, k=int(np.round(0.8 * len(chosen_images))))
        if self.mode == 'val':
            not_chosen_images = random.choices(chosen_images, k=int(np.round(0.8 * len(chosen_images))))
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
        batch_slice = range(len(self.im_paths))[index * self.batch_size: (index + 1) * self.batch_size]

        imgs_for_batch = self.slice_to_make_set(batch_slice, self.im_paths)
        y = self.slice_to_make_set(batch_slice, self.ys)

        x = np.empty((len(imgs_for_batch), *self.im_dimensions))

        for idx, i in enumerate(imgs_for_batch):
            image = cv2.imread(i)
            # x[idx] = transform(self, image, self.mode == 'train')
            x[idx] = transform(self, image, True)
        # y = to_categorical(np.array(y).astype(np.float32).reshape(-1, 1), num_classes=self.nb_classes)
        return [x, np.array(y)]


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
        ], p=1)
    transformed = transform(image=image)['image'].astype('float32')
    return transformed


def create_generators(args, seed):
    if args.dataset == 'note_styles':
        train_gen = NoteStyles(args, seed, shuffle=True, mode='train')
        val_gen = NoteStyles(args, seed, shuffle=True, mode='val')
        test_gen = NoteStyles(args, seed, shuffle=True, mode='test')

    elif args.dataset == 'cars':
        train_gen = Cars(args, seed, shuffle=True, mode='train')
        val_gen = Cars(args, seed, shuffle=True, mode='val', le=train_gen.label_encoder)
        test_gen = Cars(args, seed, shuffle=True, mode='test')
    return train_gen, val_gen, test_gen


def main():
    args = configure_parser()

    os.chdir('../data/')
    data_root = os.getcwd()
    # Dataset Loader and Sampler

    seed = np.random.choice(range(144444))

    save_path = create_save_dir(args)
    model_dir = save_path + '/untrained_model.h5'

    train_gen, val_gen, test_gen = create_generators(args, seed)

    try:
        model, criterion = create_and_compile_model(train_gen, args)
        tf.keras.models.save_model(model, model_dir)
    except urllib.error.URLError or NameError:
        print(f"Cant create from scratch, loading from {model_dir}")
        model = tf.keras.models.load_model(model_dir, custom_objects={'KerasLayer': hub.KerasLayer,
                                                                       'TF_proxy_anchor': losses.TF_proxy_anchor})
        model.compile(optimizer=tfa.optimizers.Adam(learning_rate=float(args.lr), weight_decay=args.weight_decay))

    print("Training for {} epochs.".format(args.nb_epochs))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                            filepath=save_path + '/callback_model_{epoch:02d}_{val_loss:.2f}.h5',
                                                            save_weights_only=True,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True
    )
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=save_path + '/tensorboard', histogram_freq=1)

    for epoch in range(0, args.nb_epochs):
        prepare_layers(args, epoch, model)

        pbar = tqdm(enumerate(dl_tr))
        print('###################################')
        print(f'###### TRAIN EPOCh {epoch}  #######')

        for batch_idx, (x, y) in pbar:
            x = x.numpy()
            y = y.numpy()
            x = np.moveaxis(x, 1, -1)
            model.fit(x=[x, y], batch_size=args.sz_batch, verbose=1, shuffle=False)

        pbar = tqdm(enumerate(dl_ev))
        print('###################################')
        print(f'###### TEST EPOCh {epoch}  #######')

        predict_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        for batch_idx, (x, y) in pbar:
            x = x.numpy()
            y = y.numpy()
            x = np.moveaxis(x, 1, -1)
            preds = predict_model.predict(x=[x, y], batch_size=args.sz_batch, verbose=1)

            for i in range(len(x)//args.sz_batch + 1):
                print(criterion.custom_loss(x[int(i*args.sz_batch): int((i+1)*args.sz_batch)],
                                            preds[int(i*args.sz_batch): int((i+1)*args.sz_batch)]))

        # if (epoch >= 0 and (epoch % 3 == 0)) or (epoch == args.nb_epochs - 1):
        #     test_predictions(args, epoch, model, train_gen, val_gen, test_gen)


if __name__ == '__main__':
    main()

