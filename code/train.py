import argparse, os
import utils, losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from generator import NoteStyles, Cars

import wandb


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


def create_and_compile_model():
    # model = model
    backbone = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        classes=1000,
        classifier_activation='softmax')
    flat = tf.keras.layers.Flatten()(backbone.output)
    embed = tf.keras.layers.Dense(args.sz_embedding, kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=False, activation=None)(flat)
    model = Model(inputs=backbone.input, outputs=embed)
    criterion = losses.TF_proxy_anchor(model, len(train_gen.ys), train_gen.batch_size, args.sz_embedding)
    model.compile(loss=criterion.proxy_anchor_loss, optimizer=tf.keras.optimizers.Adam(global_clipnorm=10))
    return model


args = configure_parser()

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model, args.loss, args.sz_embedding, args.alpha, 
                                                                                            args.mrg, args.optimizer, args.lr, args.sz_batch, args.remark)
# Wandb Initialization
wandb.login(key='f0a1711b34f7b07e32150c85c67697eb82c5120f')
wandb.init(project=args.dataset + '_ProxyAnchor', notes=LOG_DIR)
wandb.config.update(args)

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler

seed = np.random.choice(range(144444))
if args.dataset == 'note_styles':
    train_gen = NoteStyles(args, seed, shuffle=True, mode='train')
    val_gen   = NoteStyles(args, seed, shuffle=True, mode='val')
    test_gen  = NoteStyles(args, seed, shuffle=True, mode='test')

elif args.dataset != 'Inshop':
    train_gen = Cars(args, seed, shuffle=True, mode='train')
    val_gen = Cars(args, seed, shuffle=True, mode='val')
    test_gen = Cars(args, seed, shuffle=True, mode='test')

model = create_and_compile_model()

print("Training for {} epochs.".format(args.nb_epochs))

checkpoint_filepath = args.LOG_DIR\
                      + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model,
                                                                                            args.loss,
                                                                                            args.sz_embedding,
                                                                                            args.alpha,
                                                                                            args.mrg, args.optimizer,
                                                                                            args.lr, args.sz_batch,
                                                                                            args.remark)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


for epoch in range(0, args.nb_epochs):
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

    m = model.fit(train_gen, verbose=1, shuffle=True)

    if epoch % 3 == 0:
        print('#####################')
        print('###### TRAIN  #######')
        Recalls = utils.evaluate_cos(model, train_gen, epoch, args)

        print('#####################')
        print('######  TEST  #######')
        Recalls = utils.evaluate_cos(model, test_gen, epoch, args)







