import argparse, os
import urllib

import utils, losses
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
from generator import NoteStyles, Cars

import tensorflow_addons as tfa
import tensorflow_hub as hub


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


class L2Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(L2Layer, self).__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return K.l2_normalize(inputs, axis=1)


def create_and_compile_model(train_gen, args):
    # model = model
    y_input = Input(shape=(1,), name='Y Layer')
    x_input = Input(shape=train_gen.im_dimensions, name='Img Layer')
    backbone = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v2/feature_vector/5",
                                                  trainable=True, arguments=dict(return_endpoints=True,
                                                                                 batch_norm_momentum=0.9))(x_input)
    del backbone['default']
    del backbone['InceptionV2/global_pool']
    gap = tf.keras.layers.GlobalAveragePooling2D()(backbone['InceptionV2/Mixed_5c'])
    gmp = tf.keras.layers.GlobalMaxPooling2D()(backbone['InceptionV2/Mixed_5c'])
    add = tf.keras.layers.Add()([gap, gmp])
    #del backbone[f'InceptionV2/Logits']
    #flat = tf.keras.layers.Flatten()(backbone)
    embed = tf.keras.layers.Dense(args.sz_embedding, kernel_initializer=tf.keras.initializers.HeNormal(),
                                  use_bias=False, activation=None)(add)


    l2_norm = L2Layer()(embed)

    criterion = losses.TF_proxy_anchor(len(set(train_gen.ys)), args.sz_embedding)
    crit_tensor = criterion([y_input, l2_norm])
    model = Model(inputs=[x_input, y_input], outputs=crit_tensor)
    optimizers = [
        tfa.optimizers.AdamW(learning_rate=float(args.lr), weight_decay=args.weight_decay, clipvalue=10),
        tfa.optimizers.AdamW(learning_rate=float(args.lr)*100, weight_decay=args.weight_decay, clipvalue=10)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[0:-2]), (optimizers[1], model.layers[-2::])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(optimizer=optimizer,
                  run_eagerly=False)
    return model, criterion


def create_generators(args, seed):
    if args.dataset == 'note_styles':
        train_gen = NoteStyles(args, seed, shuffle=False, mode='train')
        val_gen = NoteStyles(args, seed, shuffle=False, mode='val')
        test_gen = NoteStyles(args, seed, shuffle=False, mode='test')

    elif args.dataset == 'cars':
        train_gen = Cars(args, seed, shuffle=False, mode='train')
        val_gen = Cars(args, seed, shuffle=False, mode='val')
        test_gen = Cars(args, seed, shuffle=False, mode='test')
    return train_gen, val_gen, test_gen


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

    print('###################################')
    print(f'######  TEST EPOCh {epoch}  #######')
    Recalls = utils.evaluate_cos(predict_model, test_gen, epoch, args)

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
            model.layers[1].trainable = False
            print(model.summary())
        if epoch == args.warm:
            model.layers[1].trainable = True
            print(model.summary())

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


def main():
    args = configure_parser()

    os.chdir('../data/')
    data_root = os.getcwd()
    # Dataset Loader and Sampler

    seed = np.random.choice(range(144444))
    train_gen, val_gen, test_gen = create_generators(args, seed)

    save_path = create_save_dir(args)
    model_dir = save_path + '/untrained_model.h5'

    try:
        model, criterion = create_and_compile_model(train_gen, args)
        tf.keras.models.save_model(model, model_dir)
    except urllib.error.URLError or NameError:
        print(f"Cant create from scratch, loading from {model_dir}")
        model = tf.keras.models.load_model(model_dir, custom_objects={'KerasLayer': hub.KerasLayer,
                                                                       'TF_proxy_anchor': losses.TF_proxy_anchor})
        model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=float(args.lr), weight_decay=args.weight_decay))

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

        predict_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        x, y = train_gen.__getitem__(0)
        m = predict_model.predict([x, y])

        model.fit(x=train_gen, validation_data=val_gen, verbose=1, shuffle=False, callbacks=[model_checkpoint_callback,
                                                                                             tensorBoard])

        if (epoch >= 0 and (epoch % 3 == 0)) or (epoch == args.nb_epochs - 1):
            test_predictions(args, epoch, model, train_gen, val_gen, test_gen)


if __name__ == '__main__':
    main()

