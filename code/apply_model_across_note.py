import os
import pickle
import random
import sys
import time
import seaborn as sns
import PIL
import cv2
import mplcursors
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
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


def load_model(args, model_name=None):
    if sys.platform != 'linux':
        notes_loc = 'D:/raw_data/1604_data/1604_notes/'
        genuine_notes_loc = 'D:/raw_data/genuines/Pack_100_4/'
        model_locations = 'D:/model_outputs/proxy_anchor/logs/'
    else:
        notes_loc = '/mnt/ssd1/Genesys_2_Capture/counterfeit/'
        genuine_notes_loc = '/mnt/ssd1/Genesys_2_Capture/genuine/100_4/'
        model_locations = '../logs/'

    for (root, dirs, files) in os.walk(model_locations):
        for dir in dirs:
            if dir == model_name:
                model_directory = sorted([f'{root}/{dir}/{i}' for i in os.listdir(f'{root}/{dir}')
                                          if os.path.isdir(f'{root}/{dir}/{i}')],
                                         key=lambda i: float(i.split('_')[-1]))[1]
                model_directory += '/'

    model = create_model(args)
    check_path = model_directory + [i for i in os.listdir(model_directory) if i.endswith('.pth')][0]
    checkpoint = torch.load(check_path)
    print(f'model directory is {check_path}')

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
    embedding = model(transformed[None, :])

    # Xhat, That, neighbors = add_pred_to_set(embedding, X, T)
    # pic, neighbors_to_pic = len(Xhat) - 1, neighbors[len(Xhat) - 1]
    #
    # y_pred_label = predict_label(Xhat, That, neighbors_to_pic, pic)
    # whole_note_label = coarse_dict[y_pred_label]
    whole_note_label = 'test'
    return whole_note_label, embedding


def load_model_stack(models):
    front_model, coarse_test_fnt, fine_test_fnt, \
        coarse_val_fnt, fine_val_fnt, _, _, front_model_dir = load_model(args, models['front'])

    args.dataset = 'note_families_back'
    back_model, coarse_test_bck, fine_test_bck, \
        coarse_val_bck, fine_val_bck, _, _, back_model_dir = load_model(args, models['back'])

    args.dataset = 'note_families_seal'
    seal_model, coarse_test_seal, fine_test_seal, \
        coarse_val_seal, fine_val_seal, notes_loc, genuine_notes_loc, seal_model_dir = load_model(args, models['seal'])

    return [front_model, front_model_dir, seal_model, seal_model_dir, back_model, back_model_dir],\
           [fine_val_fnt, coarse_val_fnt, fine_val_seal, coarse_val_seal, fine_val_bck, coarse_val_bck], \
           [fine_test_fnt, coarse_test_fnt, fine_test_seal, coarse_test_seal, fine_test_bck, coarse_test_bck], \
           notes_loc, \
           genuine_notes_loc


def count_notes(notes_per_family, genuine_notes_loc, notes_loc, args):
    total_notes = 0
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]
        if pnt_key == 'NO DATA':
            continue

        valid_notes = get_valid_notes(genuine_notes_loc, notes_loc, notes_frame, ['RGB'], ['Front'])
        total_notes += len(valid_notes)

    if args.dataset == 'note_families_tile':
        total_notes = total_notes * 6
    print(f'Found {total_notes} samples')
    return total_notes


if __name__ == '__main__':
    PLOT_IMAGES = False
    maskrcnn = MaskRCNN()

    args = parse_arguments()

    models = {'note_families_front': 'swept-pine-110',
              'note_families_back': 'dark-surf-14',
              'note_families_seal': 'major-fire-14'}
    models['note_families_tile'] = models['note_families_front']

    model, coarse_test, fine_test_fnt, \
        coarse_val_fnt, fine_val_fnt, notes_loc, genuine_notes_loc, model_dir = load_model(args, models[args.dataset])

    notes_per_family = get_notes_per_family(notes_loc, genuine_notes_loc)

    total_notes = count_notes(notes_per_family, genuine_notes_loc, notes_loc, args)

    img_inputs = []
    predictions = []
    embeddings = []
    note_labels = []
    circ_labels = []

    X_test, T_test = get_X_T('eval_', model_dir)
    X_val, T_val = get_X_T('val_', model_dir)

    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]
        if pnt_key == 'NO DATA':
            continue

        valid_notes = get_valid_notes(genuine_notes_loc, notes_loc, notes_frame, ['RGB'], ['Front'])

        if pnt_key == 'GENUINE':
            valid_notes = valid_notes[0:40]

        if len(valid_notes) > 0:
            pbar = tqdm(valid_notes, total=len(valid_notes))

            for iter, (side, spec, pack, note_num, note_dir) in enumerate(pbar):
                root_loc = f'{notes_loc}Pack_{pack}/'
                note_image, back_note_image, seal, df = get_front_back_seal(note_dir, maskrcnn)

                if args.dataset != 'note_families_tile':
                    whole_label, embedding = predict_from_image(note_image, model, X_test, T_test, False,
                                                           coarse_test)
                    predictions.append(whole_label == pnt_key)
                    embeddings.append(embedding)

                    circ_labels.append(f'PN: {pnt_key},  C: {circ_key}')
                    tiles = []
                else:
                    _, tiles, y_fac, x_fac = create_tiles(note_image)

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

                for idx, (tile, position) in enumerate(zip(tiles, ['_tl', '_bl', '_tc', '_bc', '_tr', '_br'])):
                    row_no = idx % y_fac
                    col_no = idx // y_fac

                    tile_label, embedding = predict_from_image(tile, model, X_val, T_val, False, coarse_val_fnt)
                    predictions.append(tile_label == pnt_key)
                    embeddings.append(embedding)
                    circ_labels.append(f'PN: {pnt_key},  C: {circ_key + position}')
                    note_labels.append(f'pack_{pack}_note_{note_num}')

                    if PLOT_IMAGES:
                        axs[str(idx)].imshow(tile)
                        axs[str(idx)].axis('off')
                        axs[str(idx)].title.set_text(tile_label)

                # pbar.set_description(f'{np.round(sum(whole_front_predictions) / len(whole_front_predictions), 3)}  '
                #                      f'{np.round(sum(whole_back_predictions) / len(whole_back_predictions), 3)}  '
                #                      f'{np.round(sum(whole_seal_predictions) / len(whole_seal_predictions), 3)}  '
                #                      f'{np.round(sum(tile_predictions) / len(tile_predictions), 3)}')

                if PLOT_IMAGES:
                    axs[str(6)].imshow(note_image)
                    axs[str(6)].axis('off')
                    axs[str(6)].title.set_text(f'Whole Front: Predicted: {whole_label}')

                    axs[str(7)].imshow(back_note_image)
                    axs[str(7)].axis('off')
                    axs[str(7)].title.set_text(f'Whole Back: Predicted: {whole_label}')

                    axs[str(8)].imshow(seal)
                    axs[str(8)].axis('off')
                    axs[str(8)].title.set_text(f'Whole Seal: Predicted: {whole_label}')

                    plt.suptitle(f'Whole Note: Truth: {pnt_key}')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.01, hspace=0.25)
                    #plt.show()
                    plot_dir = f'../training/{args.dataset}/{model_dir.split("/")[-2]}/plots/{os.path.splitext(os.path.split(note_dir)[-1])[0]}.png'
                    os.makedirs(f'../training/{args.dataset}/{model_dir.split("/")[-2]}/plots/', exist_ok=True)
                    print(f'Figure saved to {plot_dir}')
                    plt.savefig(plot_dir)
                    plt.close()

    print(f'{args.dataset}: {np.round(sum(predictions)/len(predictions), 3)} out of {len(predictions)} samples')

    label_array = circ_labels
    path_array = note_labels

    tsne = TSNE(n_components=2, verbose=0, perplexity=30)
    xxx = torch.cat(embeddings)
    z = tsne.fit_transform(xxx.detach().numpy())
    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    cmap = ListedColormap(sns.color_palette("husl", len(np.unique(circ_labels))).as_hex())
    colours = {pnt: cmap.colors[idx] for idx, pnt in enumerate(np.unique(circ_labels))}

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes()

    x = df["comp-1"]
    y = df["comp-2"]
    col = [colours[i] for i in list(circ_labels)]
    labels = [i for i in list(circ_labels)]

    axes_obj = ax.scatter(x,
                          y,
                          s=30,
                          c=col,
                          marker='o',
                          alpha=1
                          )
    axes_obj.annots = labels
    axes_obj.im_paths = note_labels

    plt.legend(labels=labels)

    ax.legend(bbox_to_anchor=(1.02, 1))
    mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(
        sel.artist.annots[sel.target.index]))
    mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(sel.artist.im_paths[sel.target.index]))
    # save
    fig.suptitle("TSNE")
    if sys.platform != 'linux':
        outpath = f'D:/model_outputs/proxy_anchor/applied_models/{args.dataset}/tSNE.pkl'
        os.makedirs(os.path.split(outpath)[0], exist_ok=True)
        pickle.dump(fig, open(outpath, 'wb'))
    else:
        os.makedirs(f'../applied_models/{args.dataset}/', exist_ok=True)
        pickle.dump(fig, open(f'../applied_models/{args.dataset}/tSNE.pkl', 'wb'))
    plt.close()
