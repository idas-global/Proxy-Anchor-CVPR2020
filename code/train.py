import pickle
import sys
import traceback

import cv2
import mplcursors
from PIL import ImageColor
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import pickle

import wandb
import warnings
from tqdm import *
import argparse, os
import pandas as pd
from net.resnet import *
from net.googlenet import *
import random, dataset, losses
from net.bn_inception import *
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler
from utils import get_accuracies, get_X_T_Y, f1_score_calc, create_and_save_viz_frame, confusion_matrices, \
    plot_relationships, save_metrics

def parse_arguments():
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


def get_transform(args, train, ds='default'):
    trans = dataset.utils.make_transform(
        is_train=train,
        dataset=ds,
        is_inception=(args.model == 'bn_inception')
    )
    return trans


def create_generators(args, data_root):
    if args.dataset in ['note_styles', 'note_families_front', 'note_families_back', 'note_families_seal', 'paper']:
        ds = 'notes'
    else:
        ds = 'default'

    seed = random.choice(range(21000000))
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        seed=seed,
        mode='train',
        le=None,
        transform=get_transform(args, True, ds=ds))

    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        seed=seed,
        mode='validation',
        le=dl_tr.dataset.label_encoder,
        transform=get_transform(args, True, ds=ds))

    dl_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )

    dl_ev = None
    if args.dataset not in ['note_styles']:
        ev_dataset = dataset.load(
            name=args.dataset,
            root=data_root,
            seed=None,
            mode='eval',
            le=dl_tr.dataset.label_encoder,
            transform=get_transform(args, False, ds=ds)
        )
        dl_ev = torch.utils.data.DataLoader(
            ev_dataset,
            batch_size=args.sz_batch,
            shuffle=True,
            num_workers=args.nb_workers,
            pin_memory=True
        )
    return dl_tr, dl_val, dl_ev


def create_model(args):
    # Backbone Model
    if args.model.find('googlenet') + 1:
        model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                          bn_freeze=args.bn_freeze)
    elif args.model.find('bn_inception') + 1:
        model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                             bn_freeze=args.bn_freeze)
    elif args.model.find('resnet18') + 1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
    elif args.model.find('resnet50') + 1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
    elif args.model.find('resnet101') + 1:
        model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                          bn_freeze=args.bn_freeze)
    # model = model
    if args.gpu_id == -1:
        model = nn.DataParallel(model)
    return model


def create_loss(nb_classes):
    # DML Losses
    if args.loss == 'Proxy_Anchor':
        criterion = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg,
                                        alpha=args.alpha)
    elif args.loss == 'Proxy_NCA':
        criterion = losses.Proxy_NCA(nb_classes=nb_classes, sz_embed=args.sz_embedding)
    elif args.loss == 'MS':
        criterion = losses.MultiSimilarityLoss()
    elif args.loss == 'Contrastive':
        criterion = losses.ContrastiveLoss()
    elif args.loss == 'Triplet':
        criterion = losses.TripletLoss()
    elif args.loss == 'NPair':
        criterion = losses.NPairLoss()
    return criterion


def create_optimizer_and_prepare_layers():
    # Train Parameters
    param_groups = [
        {'params': list(
            set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else
        list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {
            'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(),
            'lr': float(args.lr) * 1},
    ]
    if args.loss == 'Proxy_Anchor':
        param_groups.append({'params': criterion.parameters(), 'lr': float(args.lr) * 100})
    elif args.loss == 'Proxy_NCA':
        param_groups.append({'params': criterion.parameters(), 'lr': float(args.lr)})

    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9,
                              nesterov=True)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=0.9)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    return opt, scheduler


def freeze_layers():
    modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
    for m in modules:
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def perform_warmup(epoch):
    if args.gpu_id != -1:
        unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
    else:
        unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())
    if epoch == 0:
        for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
            param.requires_grad = False
    if epoch == args.warm:
        for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
            param.requires_grad = True


def torch_save(save_dir):
    torch.save({'model_state_dict': model.state_dict()},
               '{}/{}_{}_best.pth'.format(save_dir, args.dataset, args.model))


def text_save(recalls, best_epoch):
    with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
        f.write('Best Epoch: {}\n'.format(best_epoch))
        for key, val in recalls.items():
            f.write(f'{key} : {val}')


def save_prediction_material(save_dir, X, T, dl_tr, dl_val, dl_ev, prepend='val_'):
    np.save(f'{save_dir}/{prepend}X.npy', np.array(X))
    np.save(f'{save_dir}/{prepend}T.npy', np.array(T))
    if prepend == 'val_':
        np.save(f'{save_dir}/{prepend}image_paths.npy', np.array(dl_tr.dataset.im_paths + dl_val.dataset.im_paths))
    else:
        np.save(f'{save_dir}/{prepend}image_paths.npy', np.array(dl_ev.dataset.im_paths))

    dl_list = [dl_tr, dl_val]
    if dl_ev:
        dl_list = [dl_tr, dl_val, dl_ev]

    for dataloader in dl_list:
        with open(f'{save_dir}/{dataloader.dataset.mode}_coarse_dict.pkl', 'wb') as f:
            pickle.dump(dataloader.dataset.class_names_coarse_dict, f)

        with open(f'{save_dir}/{dataloader.dataset.mode}_fine_dict.pkl', 'wb') as f:
            pickle.dump(dataloader.dataset.class_names_fine_dict, f)


def train_model(args, model, dl_tr, dl_val, dl_ev):
    losses_list = []
    key_to_opt = f'eval_f1score@7'
    best_recall = pd.DataFrame()
    best_recall[key_to_opt] = [0]

    for epoch in range(0, args.nb_epochs):
        model.train()
        bn_freeze = args.bn_freeze
        if bn_freeze:
            freeze_layers()

        losses_per_epoch = []

        # Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            perform_warmup(epoch)

        pbar = tqdm(enumerate(dl_tr))

        for batch_idx, (x, y) in pbar:
            if batch_idx > 0:
                break
            run_batch(batch_idx, dl_tr, epoch, losses_per_epoch, model, pbar, x, y)

        losses_list.append(np.mean(losses_per_epoch))
        wandb.log({'loss': losses_list[-1]}, step=epoch)
        scheduler.step()

        if epoch >= 0 and (epoch % np.floor(args.nb_epochs/5) == 0 or epoch == args.nb_epochs - 1):
            with torch.no_grad():
                save_dir = '{}/{}/{}_{}'.format(LOG_DIR, wandb.run.name, wandb.run.name, np.round(best_recall[key_to_opt].values[0], 3))
                os.makedirs(save_dir, exist_ok=True)

                val_recalls, X, T = evaluate_cos(model, dl_tr, epoch, args, validation=dl_val)

                save_prediction_material(save_dir, X, T, dl_tr, dl_val, dl_ev)
                post_to_wandb(epoch, val_recalls)

                if dl_ev:
                    test_recalls, X, T = evaluate_cos(model, dl_ev, epoch, args)
                    save_prediction_material(save_dir, X, T, dl_tr, dl_val, dl_ev, prepend='eval_')

                    post_to_wandb(epoch, test_recalls, postpend='_test')
                else:
                    test_recalls = val_recalls
                    key_to_opt = 'validation_f1score@7'
                    if key_to_opt not in best_recall.keys():
                        best_recall[key_to_opt] = [0]

            torch_save(save_dir)

            # Best model save
            if best_recall[key_to_opt].values[0] < test_recalls[key_to_opt].values[0]:
                best_recall, best_epoch = test_recalls, epoch

            text_save(test_recalls, best_epoch)



def post_to_wandb(epoch, val_recalls, postpend='_validation'):
    for key, val in val_recalls.items():
        wandb.log({key + postpend: val.values[0]}, step=epoch)
        print(f'{key} : {np.round(val.values[0], 3)}')


def run_batch(batch_idx, dl_tr, epoch, losses_per_epoch, model, pbar, x, y):
    m = model(x.squeeze())
    loss = criterion(m, y.squeeze())

    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
    losses_per_epoch.append(loss.data.cpu().numpy())

    opt.step()
    pbar.set_description(
        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            epoch, batch_idx + 1, len(dl_tr), 100. * batch_idx / len(dl_tr),
            loss.item()))


def plot_tSNE(X, data_viz_frame, dataloader, pictures_to_predict, train_dest, val_dest, test_dest):
    tsne = TSNE(n_components=2, verbose=0, perplexity=dataloader.dataset.perplex)
    z = tsne.fit_transform(X[pictures_to_predict])
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    params = ['truth']
    degrees = ['fine']
    for deg in degrees:
        for para in params:
            import seaborn as sns
            cmap = ListedColormap(sns.color_palette("husl", len(np.unique(data_viz_frame[f"{para}_label_{deg}"]))).as_hex())
            colours = {pnt: cmap.colors[idx] for idx, pnt in enumerate(np.unique(data_viz_frame[f"{para}_label_{deg}"]))}

            fig = plt.figure(figsize=(12, 12))
            ax = plt.axes()

            x = df["comp-1"]
            y = df["comp-2"]
            col = [colours[i] for i in list(data_viz_frame[f"{para}_label_{deg}"].values)]
            labels = [i for i in list(data_viz_frame[f"{para}_label_{deg}"].values)]

            axes_obj = ax.scatter(x,
                                  y,
                                  s=30,
                                  c=col,
                                  marker='o',
                                  alpha=1
                                )
            axes_obj.annots = labels
            axes_obj.labels = labels
            if dataloader.dataset.mode != 'validation':
                axes_obj.im_paths = list(np.array(dataloader.dataset.tsne_labels)[pictures_to_predict])
            else:
                axes_obj.im_paths = list(np.array(dataloader.dataset.tsne_labels))

            mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(sel.artist.annots[sel.target.index]))
            mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(sel.artist.im_paths[sel.target.index]))
            # save
            fig.suptitle("TSNE")
            if dataloader.dataset.mode == 'train':
                pickle.dump(fig, open(f'{train_dest}{para}_{deg}_tSNE.pkl', 'wb'))
            if dataloader.dataset.mode == 'validation':
                pickle.dump(fig, open(f'{val_dest}{para}_{deg}_tSNE.pkl', 'wb'))
            if dataloader.dataset.mode == 'eval':
                pickle.dump(fig, open(f'{test_dest}{para}_{deg}_tSNE.pkl', 'wb'))
            plt.close()

    # fig2 = pickle.load(open('FigureObject.fig.pickle', 'rb'))
    # mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(sel.artist.annots[sel.target.index]))
    # mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(sel.artist.im_paths[sel.target.index]))
    # fig2.show()


def evaluate_cos(model, dataloader, epoch, args, validation=None):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model and get targets
    test_dest = f'../training/{args.dataset}/{wandb.run.name}/{epoch}/test/'
    val_dest = f'../training/{args.dataset}/{wandb.run.name}/{epoch}/validation/'
    train_dest = f'../training/{args.dataset}/{wandb.run.name}/{epoch}/train_and_validation/'

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    X, T, Y, neighbors = get_X_T_Y(dataloader, model, validation) # X: Embeddings, T: True Labels,
                                                                  # Y: True Labels of neighbors

    pictures_to_predict = np.sort(random.choices(range(len(X)), k=int(round(len(X)*50/100))))
    dl_loader = dataloader
    if validation is not None:
        pictures_to_predict = list(range(len(X) - len(validation.dataset.ys), len(X)))
        dl_loader = validation

    metrics = f1_score_calc(T, Y, dl_loader, pictures_to_predict)
    coarse_filter_dict, fine_filter_dict, y_preds, y_true = get_accuracies(T, X, dl_loader,
                                                                           neighbors, pictures_to_predict, metrics)

    data_viz_frame = create_and_save_viz_frame(X, dl_loader, coarse_filter_dict, fine_filter_dict,
                                               pictures_to_predict,
                                               train_dest, val_dest, test_dest,
                                               y_preds, y_true)

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     params = ['prediction', 'truth']
    #     degrees = ['fine', 'coarse']
    #     for deg in degrees:
    #         for para in params:
    #             try:
    #                 plot_relationships(X, data_viz_frame, dl_loader,
    #                                deg, para, pictures_to_predict,
    #                                train_dest, val_dest, test_dest)
    #             except Exception:
    #                 pass
    try:
        plot_tSNE(X, data_viz_frame, dl_loader, pictures_to_predict, train_dest, val_dest, test_dest)
    except Exception:
        print(traceback.format_exc())

    # try:
    #     confusion_matrices(data_viz_frame, dataloader, train_dest, val_dest, test_dest)
    # except Exception:
    #     pass

    save_metrics(dataloader, metrics, train_dest, val_dest, test_dest)
    model.train()
    model.train(model_is_training)  # revert to previous training state
    return metrics, X, T


def test_generator_labels(dl_tr, dl_val, dl_ev):
    dl_tr_labels = sorted(dict(Counter(dl_tr.dataset.class_names_fine)).items(), key=lambda x: x[0])
    dl_val_labels = sorted(dict(Counter(dl_val.dataset.class_names_fine)).items(), key=lambda x: x[0])
    if dl_ev:
        dl_ev_labels = sorted(dict(Counter(dl_ev.dataset.class_names_fine)).items(), key=lambda x: x[0])
        print('##### TEST LABELS #####')
        for v in dl_ev_labels:
            print(f'{v}')
        print('#######################')

    if _train_val_labels_different(dl_tr_labels, dl_val_labels):
        sys.exit()

    for t, v in zip(dl_tr_labels, dl_val_labels):
        print(f'{t}      {v}')

    assert dl_tr.dataset.class_names_coarse_dict == dl_val.dataset.class_names_coarse_dict
    assert dl_tr.dataset.class_names_fine_dict == dl_val.dataset.class_names_fine_dict

    if sys.platform != 'linux':
        dl_list = [dl_tr, dl_val]
        if dl_ev:
            dl_list = [dl_ev, dl_tr, dl_val]

        for dataloader in dl_list:
            for i in random.choices(range(len(dataloader.dataset.im_paths)), k=15):
                x, y = dataloader.dataset.__getitem__(i)
                fig, axs = plt.subplots(1, 2)
                HWC = np.moveaxis(np.array(x), 0, -1)
                axs[0].imshow(HWC)
                axs[1].imshow(((HWC - HWC.min()) * (1/(HWC.max() - HWC.min()) * 255)).astype('uint8'))
                plt.title(f'Sample from {dataloader.dataset.mode} : {dataloader.dataset.class_names_coarse_dict[y]}')
                plt.suptitle(dataloader.dataset.class_names_fine_dict[y])
                plt.show()

        for i in random.choices(range(len(dl_tr.dataset.im_paths)), k=15):
            train_y = dl_tr.dataset.ys[i]
            plt.imshow(cv2.imread(dl_tr.dataset.im_paths[i]))
            plt.title(dl_tr.dataset.class_names_fine_dict[train_y])
            plt.suptitle(dl_tr.dataset.class_names_coarse_dict[train_y])
            plt.show()

            val_idx = list(dl_val.dataset.ys).index(train_y)

            plt.imshow(cv2.imread(dl_val.dataset.im_paths[val_idx]))
            plt.title(dl_val.dataset.class_names_fine_dict[train_y])
            plt.suptitle(dl_val.dataset.class_names_coarse_dict[train_y])
            plt.show()


def _train_val_labels_different(dl_tr_labels, dl_val_labels):
    terminate = False
    try:
        assert len(dl_tr_labels) == len(dl_val_labels)
    except AssertionError:
        for i in dl_tr_labels:
            print(i)
        print('---------------')
        for i in dl_val_labels:
            print(i)
        terminate = True
    return terminate


if __name__ == '__main__':
    args = parse_arguments()

    model = create_model(args)

    # Directory for Log
    LOG_DIR = args.LOG_DIR + '/{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset,
                                                                                            args.model,
                                                                                            args.loss,
                                                                                            args.sz_embedding,
                                                                                            args.alpha,
                                                                                            args.mrg,
                                                                                            args.optimizer,
                                                                                            args.lr, args.sz_batch,
                                                                                            args.remark)
    # Wandb Initialization
    wandb.login(key='f0a1711b34f7b07e32150c85c67697eb82c5120f')
    wandb.init(project=args.dataset + '_ProxyAnchor', notes=LOG_DIR)
    wandb.config.update(args)

    os.chdir('../data/')
    data_root = os.getcwd()

    dl_tr, dl_val, dl_ev = create_generators(args, data_root)
    test_generator_labels(dl_tr, dl_val, dl_ev)

    nb_classes = dl_tr.dataset.nb_classes()

    criterion = create_loss(nb_classes)
    opt, scheduler = create_optimizer_and_prepare_layers()

    print("Training for {} epochs.".format(args.nb_epochs))
    train_model(args, model, dl_tr, dl_val, dl_ev)
