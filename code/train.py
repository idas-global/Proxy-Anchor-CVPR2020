import argparse, os
import random, dataset, utils, losses
from collections import Counter

import pandas as pd

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


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


def get_transform(train):
    trans = dataset.utils.make_transform(
        is_train=train,
        is_inception=(args.model == 'bn_inception')
    )
    return trans


def create_generators():
    seed = random.choice(range(21000000))
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        seed=seed,
        mode='train',
        le=None,
        transform=get_transform(True))

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
        mode='train',
        le=dl_tr.dataset.label_encoder,
        transform=get_transform(True))
    dl_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )

    dl_ev = None
    if args.dataset not in ['note_styles', 'note_families_front', 'note_families_back', 'note_families_seal']:
        ev_dataset = dataset.load(
            name=args.dataset,
            root=data_root,
            seed=None,
            mode='eval',
            le=dl_tr.dataset.label_encoder,
            transform=get_transform(False)
        )
        dl_ev = torch.utils.data.DataLoader(
            ev_dataset,
            batch_size=args.sz_batch,
            shuffle=False,
            num_workers=args.nb_workers,
            pin_memory=True
        )
    print(dict(Counter(dl_tr.dataset.class_names_coarse)))
    print(dict(Counter(dl_val.dataset.class_names_coarse)))
    if dl_ev:
        print(dict(Counter(dl_ev.dataset.class_names_coarse)))

    # le_name_mapping_train = dict(zip(dl_tr.dataset.label_encoder.classes_,
    #                            dl_tr.dataset.label_encoder.transform(dl_tr.dataset.label_encoder.classes_)))
    #
    # le_name_mapping_val = dict(zip(dl_val.dataset.label_encoder.classes_,
    #                             dl_val.dataset.label_encoder.transform(dl_val.dataset.label_encoder.classes_)))
    #
    # assert le_name_mapping_train == le_name_mapping_val
    # import matplotlib.pyplot as plt
    # import cv2
    # for i in random.choices(range(len(dl_tr.dataset.im_paths)), k=5):
    #     train_y = dl_tr.dataset.ys[i]
    #     plt.imshow(cv2.imread(dl_tr.dataset.im_paths[i]))
    #     plt.title(dl_tr.dataset.class_names_fine[i])
    #     plt.suptitle(dl_tr.dataset.class_names_coarse_dict[train_y])
    #     plt.show()
    #
    #     assert train_y == le_name_mapping_train[dl_tr.dataset.class_names_fine[i]]
    #
    #     val_idx = list(dl_val.dataset.ys).index(train_y)
    #     assert train_y == le_name_mapping_val[dl_val.dataset.class_names_fine[val_idx]]
    #
    #     plt.imshow(cv2.imread(dl_val.dataset.im_paths[val_idx]))
    #     plt.title(dl_val.dataset.class_names_fine[val_idx])
    #     plt.suptitle(dl_val.dataset.class_names_coarse_dict[train_y])
    #     plt.show()
    #
    #     if dl_ev is not None:
    #         assert train_y not in list(dl_ev.dataset.ys)
    return dl_tr, dl_val, dl_ev


def create_model():
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
    # Optimizer Setting
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
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()},
               '{}/{}_{}_best.pth'.format(save_dir, args.dataset, args.model))


def text_save(recalls, best_epoch):
    with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
        f.write('Best Epoch: {}\n'.format(best_epoch))
        for key, val in recalls.items():
            f.write(f'{key} : {val}')


def train_model(args, model, dl_tr, dl_val, dl_ev):
    losses_list = []
    k = 7
    best_recall = pd.DataFrame()
    best_recall[f'f1score@{k}'] = [0]

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
            m = model(x.squeeze())
            loss = criterion(m, y.squeeze())

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

            losses_per_epoch.append(loss.data.cpu().numpy())
            opt.step()

            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(dl_tr),
                           100. * batch_idx / len(dl_tr),
                    loss.item()))

        losses_list.append(np.mean(losses_per_epoch))
        wandb.log({'loss': losses_list[-1]}, step=epoch)
        scheduler.step()

        if epoch > 0 and (epoch % 5 == 0 or epoch == args.nb_epochs - 1):
            with torch.no_grad():
                if dl_ev:
                    test_recalls = utils.evaluate_cos(model, dl_ev, epoch, args)

                    for key, val in test_recalls.items():
                        wandb.log({key + '_test': val.values[0]}, step=epoch)
                        print(f'{key} : {np.round(val.values[0], 3)}')

                val_recalls = utils.evaluate_cos(model, dl_tr, epoch, args, validation=dl_val)

                for key, val in val_recalls.items():
                    wandb.log({key + '_validation': val.values[0]}, step=epoch)
                    print(f'{key} : {np.round(val.values[0], 3)}')

            # Best model save
            if best_recall[f"f1score@{k}"].values[0] < val_recalls[f"f1score@{k}"].values[0]:
                best_recall = val_recalls
                best_epoch = epoch

                save_dir = '{}/{}_{}'.format(LOG_DIR, wandb.run.name, np.round(best_recall[f"f1score@{k}"].values[0], 3))
                torch_save(save_dir)
                text_save(val_recalls, best_epoch)


if __name__ == '__main__':
    args = parse_arguments()

    model = create_model()

    # Directory for Log
    LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset,
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

    dl_tr, dl_val, dl_ev = create_generators()
    nb_classes = dl_tr.dataset.nb_classes()

    criterion = create_loss(nb_classes)
    opt, scheduler = create_optimizer_and_prepare_layers()

    print("Training for {} epochs.".format(args.nb_epochs))
    train_model(args, model, dl_tr, dl_val, dl_ev)
