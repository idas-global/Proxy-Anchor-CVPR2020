import os
import pickle
import random
import sys
import time

import PIL
import cv2
import mplcursors
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from augment_paper import get_valid_notes, get_front_back_seal, form_genuine_frame, form_1604_frame, augment, \
    get_notes_per_family
from maskrcnn import MaskRCNN
from utils import l2_norm
from dataset.utils import RGBToBGR, ScaleIntensities, Identity
from noteclasses import ImageBMP
from train import parse_arguments, create_model, create_generators, get_transform
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd


def transform(im, train=True):
    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
        [
            RGBToBGR(),
            transforms.Resize((inception_sz_resize, inception_sz_resize * 2)) if not train else Identity(),
            transforms.Resize((inception_sz_crop, inception_sz_crop)) if train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            transforms.Normalize(mean=inception_mean, std=inception_std)
        ])
    return inception_transform(im)


def predict_label(X, T, neighbors_to_pic, pic):
    preds, counts = np.unique(T[neighbors_to_pic[0:7]], return_counts=True)
    close_preds = preds[counts >= np.max(counts) - 1]
    predictions = {}
    for close_pred in close_preds:
        neighbors_pred = [i for i in neighbors_to_pic[0:7] if T[i] == close_pred]
        one = np.array(X[neighbors_pred])
        two = np.array(X[pic]).reshape(-1, 1).T

        norms_X = np.sqrt((one * one).sum(axis=1))  # Or np.linalg.norm(axis=1)
        one /= norms_X[:, np.newaxis]
        norms_Y = np.sqrt((two * two).sum(axis=1))
        two /= norms_Y[:, np.newaxis]
        cs = np.dot(one, two.T)

        predictions[close_pred] = (sum(cs) / np.sqrt(len(cs)))[0]
    y_pred_label = max(predictions, key=predictions.get)
    return y_pred_label


def create_tiles(im):
    tiles = []
    pictures = 3
    edges = np.linspace(0, im.shape[1], pictures + 1)
    for start, end in zip(edges[:-1], edges[1:]):
        box = (start, 0, end, im.shape[0]/2)
        tiles.append(im[int(round(box[1])):int(round(box[3])), int(round(box[0])):int(round(box[2]))])
        box = (start, im.shape[0] / 2, end, im.shape[0])
        tiles.append(im[int(round(box[1])):int(round(box[3])), int(round(box[0])):int(round(box[2]))])

    return note_dir, tiles, 2, 3


def get_transformed_image(tile, train=True):
    im = PIL.Image.fromarray(tile)
    # convert gray to rgb
    if len(list(im.split())) == 1: im = im.convert('RGB')
    trans = get_transform(args, False, ds='notes')
    return trans(im)


def add_pred_to_set(y_pred, X, T):
    X = np.vstack((X, y_pred.detach().numpy()))
    X = l2_norm(X)
    X = torch.from_numpy(X)

    T = np.hstack((T, -1))
    T = torch.from_numpy(T)

    K = min(32, len(X) - 1)

    cos_sim = F.linear(X, X)
    neighbors = cos_sim.topk(1 + K)[1][:, 1:]
    neighbors = np.array(neighbors)
    return X, T, neighbors


def get_X_T(set, model_directory):
    X = np.load(f'{model_directory}/{set}X.npy')
    T = np.load(f'{model_directory}/{set}T.npy')
    return X, T


def load_model(args, model_directory=None):
    if sys.platform != 'linux':
        notes_loc = 'D:/1604_notes/'
        genuine_notes_loc = 'D:/genuines/Pack_100_4/'
        if model_directory is None:
            model_directory = '../logs/logs_note_families_front/bn_inception_Proxy_Anchor_embedding512_alpha32_mrg0.1_adamw_lr0.0001_batch32/noble-haze-96_94.049/'
    else:
        notes_loc = '/mnt/ssd1/Genesys_2_Capture/counterfeit/'
        genuine_notes_loc = '/mnt/ssd1/Genesys_2_Capture/genuine/100_4/'
        LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}/'.format(args.dataset,
                                                                                                      args.model,
                                                                                                      args.loss,
                                                                                                      args.sz_embedding,
                                                                                                      args.alpha,
                                                                                                      args.mrg,
                                                                                                      args.optimizer,
                                                                                                      args.lr,
                                                                                                      args.sz_batch,
                                                                                                      args.remark)
        if os.path.exists(LOG_DIR):
            model_directory = sorted([LOG_DIR + i for i in os.listdir(LOG_DIR) if os.path.isdir(LOG_DIR + i)],
                                     key=lambda i: float(i.split('_')[-1]))[-3]
            model_directory += '/'
            print(f'model directory is {model_directory}')
        else:
            print(f'{LOG_DIR} does not exist')
            sys.exit()
    model = create_model(args)
    checkpoint = torch.load(model_directory + [i for i in os.listdir(model_directory) if i.endswith('.pth')][0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model_is_training = model.training
    model.eval()
    with open(f'{model_directory}eval_coarse_dict.pkl', 'rb') as f:
        coarse_filter_dict_saved = pickle.load(f)
    with open(f'{model_directory}eval_fine_dict.pkl', 'rb') as f:
        fine_filter_dict_saved = pickle.load(f)
    with open(f'{model_directory}validation_coarse_dict.pkl', 'rb') as f:
        coarse_filter_dict_saved_val = pickle.load(f)
    with open(f'{model_directory}validation_fine_dict.pkl', 'rb') as f:
        fine_filter_dict_saved_val = pickle.load(f)

    return model, coarse_filter_dict_saved, fine_filter_dict_saved, coarse_filter_dict_saved_val, fine_filter_dict_saved_val, notes_loc, genuine_notes_loc, model_directory


def predict_from_image(note_image, model, X, T, train, coarse_dict):
    transformed = get_transformed_image(note_image, train=train)
    y_pred = model(transformed[None, :])

    Xhat, That, neighbors = add_pred_to_set(y_pred, X, T)
    pic, neighbors_to_pic = len(Xhat) - 1, neighbors[len(Xhat) - 1]

    y_pred_label = predict_label(Xhat, That, neighbors_to_pic, pic)
    whole_note_label = coarse_dict[y_pred_label]
    return whole_note_label


def load_model_stack():
    front_model, coarse_test_fnt, fine_test_fnt, \
        coarse_val_fnt, fine_val_fnt, _, _, front_model_dir = load_model(args)

    args.dataset = 'note_families_back'
    back_model, coarse_test_bck, fine_test_bck, \
        coarse_val_bck, fine_val_bck, _, _, back_model_dir = load_model(args)

    args.dataset = 'note_families_seal'
    seal_model, coarse_test_seal, fine_test_seal, \
        coarse_val_seal, fine_val_seal, notes_loc, genuine_notes_loc, seal_model_dir = load_model(args)

    return [front_model, front_model_dir, seal_model, seal_model_dir, back_model, back_model_dir],\
           [fine_val_fnt, coarse_val_fnt, fine_val_seal, coarse_val_seal, fine_val_bck, coarse_val_bck], \
           [fine_test_fnt, coarse_test_fnt, fine_test_seal, coarse_test_seal, fine_test_bck, coarse_test_bck], \
           notes_loc, \
           genuine_notes_loc


if __name__ == '__main__':
    PLOT_IMAGES = False
    maskrcnn = MaskRCNN()

    args = parse_arguments()

    [front_model, front_model_dir, seal_model, seal_model_dir, back_model, back_model_dir], \
    [fine_val_fnt, coarse_val_fnt, fine_val_seal, coarse_val_seal, fine_val_bck, coarse_val_bck], \
    [fine_test_fnt, coarse_test_fnt, fine_test_seal, coarse_test_seal, fine_test_bck, coarse_test_bck], \
        notes_loc, \
        genuine_notes_loc = load_model_stack()

    notes_per_family = get_notes_per_family(notes_loc, genuine_notes_loc)

    whole_front_predictions = []
    whole_back_predictions = []
    whole_seal_predictions = []
    tile_predictions = []

    X_test_fnt, T_test_fnt = get_X_T('eval_', front_model_dir)
    X_test_bck, T_test_bck = get_X_T('eval_', back_model_dir)
    X_test_seal, T_test_seal = get_X_T('eval_', seal_model_dir)

    X_val_fnt, T_val_fnt = get_X_T('val_', front_model_dir)
    X_val_bck, T_val_bck = get_X_T('val_', back_model_dir)
    X_val_seal, T_val_seal = get_X_T('val_', seal_model_dir)

    # args.dataset = 'note_families_front'
    # front_model, coarse_test_fnt, fine_test_fnt, \
    # coarse_val_fnt, fine_val_fnt, _, _, front_model_dir = load_model(args, 'D:/models/front/earnest-jazz-93_91.049/')
    # X_test_fnt, T_test_fnt = get_X_T('eval_', front_model_dir)
    #
    # from sklearn.manifold import TSNE
    # import seaborn as sns
    #
    # tsne = TSNE(n_components=2, verbose=1, perplexity=69,  early_exaggeration=12)
    # z = tsne.fit_transform(X_test_fnt)
    # df = pd.DataFrame()
    # df["y"] = [fine_test_fnt[i] for i in T_test_fnt]
    # df["comp-1"] = z[:, 0]
    # df["comp-2"] = z[:, 1]
    # df.sort_values(by=['y'])
    #
    # cmap = ListedColormap(sns.color_palette("husl", len(np.unique(df["y"]))).as_hex())
    # colours = {pnt: cmap.colors[idx] for idx, pnt in enumerate(np.unique(df["y"]))}
    #
    # fig = plt.figure(figsize=(12, 12))
    # ax = plt.axes()
    # for pnt in np.unique(df["y"]):
    #     pnt_bool = [pnt == ii for ii in df["y"]]
    #     ax.scatter(df.loc[pnt_bool, "comp-1"],
    #                df.loc[pnt_bool, "comp-2"],
    #                s=30, c=colours[pnt], marker='o', alpha=1, label=pnt)
    # ax.legend(bbox_to_anchor=(1.02, 1))
    # mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    # # save
    # fig.suptitle("TSNE")
    # plt.show()
    #
    #
    #
    #
    #
    #
    # fig = plt.figure()
    # ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 data=df).set(title="Family Projection")
    # # label points on the plot
    # ys_done = []
    # for x, y, z in zip(df["comp-1"], df["comp-2"], df['y']):
    #     if z in ys_done:
    #         continue
    #     ys_done.append(z)
    #     # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
    #     plt.text(x=x,  # x-coordinate position of data label
    #              y=y,
    #              s=z,  # data label, formatted to ignore decimals
    #              color = 'purple')  # set colour of line
    # fig.show()

    img_inputs = []
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]
        if pnt_key == 'NO DATA':
            continue

        valid_notes = get_valid_notes(genuine_notes_loc, notes_loc, notes_frame, ['RGB'], ['Front'])

        if len(valid_notes) > 0:
            pbar = tqdm(valid_notes, total=valid_notes)

            for iter, (side, spec, pack, note_num, note_dir) in enumerate(pbar):
                root_loc = f'{notes_loc}Pack_{pack}/'
                note_image, back_note_image, seal, df = get_front_back_seal(note_dir, maskrcnn)

                _, tiles, y_fac, x_fac = create_tiles(note_image)

                whole_front_label = predict_from_image(note_image, front_model, X_test_fnt, T_test_fnt, False,
                                                       coarse_test_fnt)
                whole_front_predictions.append(whole_front_label == pnt_key)


                whole_back_label = predict_from_image(back_note_image, back_model, X_test_bck, T_test_bck, False,
                                                      coarse_test_bck)
                whole_back_predictions.append(whole_back_label == pnt_key)


                whole_seal_label = predict_from_image(seal, seal_model, X_test_seal, T_test_seal, False,
                                                      coarse_test_seal)
                whole_seal_predictions.append(whole_seal_label == pnt_key)

                if PLOT_IMAGES:
                    fig, axs = plt.subplot_mosaic(
                        """
                        024
                        135
                        666
                        777
                        888
                        """
                    )

                for idx, tile in enumerate(tiles):
                    row_no = idx % y_fac
                    col_no = idx // y_fac

                    im = get_transformed_image(tile)

                    y_pred = front_model(im[None, :])
                    Xhat, That, neighbors = add_pred_to_set(y_pred, X_val_fnt, T_val_fnt)

                    pic = len(Xhat) - 1
                    neighbors_to_pic = neighbors[pic]

                    y_pred_label = predict_label(Xhat, That, neighbors_to_pic, pic)
                    tile_predictions.append(coarse_val_fnt[y_pred_label] == pnt_key)

                    if PLOT_IMAGES:
                        axs[str(idx)].imshow(tile)
                        axs[str(idx)].axis('off')
                        axs[str(idx)].title.set_text(coarse_val_fnt[y_pred_label])

                pbar.set_description(f'{np.round(sum(whole_front_predictions) / len(whole_front_predictions), 3)}  '
                                     f'{np.round(sum(whole_back_predictions) / len(whole_back_predictions), 3)}  '
                                     f'{np.round(sum(whole_seal_predictions) / len(whole_seal_predictions), 3)}  '
                                     f'{np.round(sum(tile_predictions) / len(tile_predictions), 3)}')

                if PLOT_IMAGES:
                    axs[str(6)].imshow(note_image)
                    axs[str(6)].axis('off')
                    axs[str(6)].title.set_text(f'Whole Front: Predicted: {whole_front_label}')

                    axs[str(7)].imshow(back_note_image)
                    axs[str(7)].axis('off')
                    axs[str(7)].title.set_text(f'Whole Back: Predicted: {whole_back_label}')

                    axs[str(8)].imshow(seal)
                    axs[str(8)].axis('off')
                    axs[str(8)].title.set_text(f'Whole Seal: Predicted: {whole_seal_label}')

                    plt.suptitle(f'Whole Note: Truth: {pnt_key}')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.01, hspace=0.25)
                    #plt.show()
                    plot_dir = f'../training/{args.dataset}/{front_model_dir.split("/")[-2]}/plots/{os.path.splitext(os.path.split(note_dir)[-1])[0]}.png'
                    os.makedirs(f'../training/{args.dataset}/{front_model_dir.split("/")[-2]}/plots/', exist_ok=True)
                    print(f'Figure saved to {plot_dir}')
                    plt.savefig(plot_dir)
                    plt.close()

    print(f'Front: {np.round(sum(whole_front_predictions)/len(whole_front_predictions), 3)} out of {len(whole_front_predictions)} samples')
    print(f'Back: {np.round(sum(whole_back_predictions)/len(whole_back_predictions), 3)} out of {len(whole_back_predictions)} samples')
    print(f'Seal: {np.round(sum(whole_seal_predictions)/len(whole_seal_predictions), 3)} out of {len(whole_seal_predictions)} samples')
    print(f'tile: {np.round(sum(tile_predictions)/len(tile_predictions), 3)} out of {len(tile_predictions)} samples')


'''
targets = []
images = []
for i in tqdm(range(len(dl_ev.dataset.ys) // 2), desc='Testing Eval Set'):
    image, y = dl_ev.dataset.__getitem__(i)
    y = [key for key, val in fine_filter_dict_saved.items() if val == fine_filter_dict[y]]
    if y:
        y = y[0]
    else:
        continue
    embedding = model(image[None, :])
    images.append(embedding)
    targets.append(y)

embeddings = torch.cat(images)

set = 'eval_'
X = np.load(f'{model_directory}/{set}X.npy')
train = len(X)
X = np.vstack((X, embeddings.detach().numpy()))
X = l2_norm(X)
X = torch.from_numpy(X)

T = np.load(f'{model_directory}/{set}T.npy')
T = np.hstack((T, targets))
T = torch.from_numpy(T)

K = min(64, len(X) - 1)

cos_sim = F.linear(X, X)
neighbors = cos_sim.topk(1 + K)[1][:, 1:]
neighbors = np.array(neighbors)

shuffled_idx = [train + i for i in range(len(embeddings))]
random.shuffle(shuffled_idx)
ans = []
for pic in shuffled_idx:
    neighbors_to_pic = np.array(neighbors[pic, :][~np.in1d(neighbors[pic, :], shuffled_idx)])
    whole_note_label = predict_label(X, T, neighbors_to_pic, pic)
    ans.append(coarse_filter_dict_saved[T[pic].item()] == coarse_filter_dict_saved[whole_note_label])
print(sum(ans) / len(ans))

ans = []
for i in tqdm(range(len(dl_ev.dataset.ys) // 2, len(dl_ev.dataset.ys)), desc='Testing Eval Set'):
    image, y = dl_ev.dataset.__getitem__(i)
    y = [key for key, val in fine_filter_dict_saved.items() if val == fine_filter_dict[y]]
    if y:
        y = y[0]
    else:
        continue

    embedding = model(image[None, :])
    Xhat, That, neighbors = add_pred_to_set(embedding, X, T)
    pic = len(Xhat) - 1
    neighbors_to_pic = neighbors[pic]
    whole_note_label = predict_label(Xhat, That, neighbors_to_pic, pic)
    ans.append(coarse_filter_dict_saved[y] == coarse_filter_dict_saved[whole_note_label])
print(sum(ans) / len(ans))

ans = []
X, T = get_X_T('eval_')
for i in tqdm(range(len(dl_ev.dataset.ys) // 2, len(dl_ev.dataset.ys)), desc='Testing Eval Set'):
    image, y = dl_ev.dataset.__getitem__(i)
    y = [key for key, val in fine_filter_dict_saved.items() if val == fine_filter_dict[y]]
    if y:
        y = y[0]
    else:
        continue
    embedding = model(image[None, :])
    Xhat, That, neighbors = add_pred_to_set(embedding, X, T)
    pic = len(Xhat) - 1
    neighbors_to_pic = neighbors[pic]
    whole_note_label = predict_label(Xhat, That, neighbors_to_pic, pic)
    ans.append(coarse_filter_dict_saved[y] == coarse_filter_dict_saved[whole_note_label])
print(sum(ans) / len(ans))

X, T, neighbors = add_pred_to_set(y_pred, set='eval_')
pic = len(X) - 1
neighbors_to_pic = neighbors[pic]
whole_note_label = predict_label()
whole_note_predictions.append(coarse_filter_dict[whole_note_label] == pnt_key)
'''