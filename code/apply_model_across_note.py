import os
import pickle
import sys
import time

import PIL
import cv2
import numpy as np
import torch
from tqdm import tqdm

from augment_paper import get_valid_notes, get_front_back_seal, form_genuine_frame, form_1604_frame, augment
from maskrcnn import MaskRCNN
from utils import l2_norm
from dataset.utils import RGBToBGR, ScaleIntensities, Identity
from noteclasses import ImageBMP
from train import parse_arguments, create_model, create_generators
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


def predict_label():
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
    im = transform(im, train)
    return im


def add_pred_to_set(y_pred, set='val_'):
    X = np.load(f'{model_directory}/{set}X.npy')
    X = np.vstack((X, y_pred.detach().numpy()))
    X = l2_norm(X)
    X = torch.from_numpy(X)

    T = np.load(f'{model_directory}/{set}T.npy')
    T = np.hstack((T, -1))
    T = torch.from_numpy(T)

    K = min(32, len(X) - 1)

    cos_sim = F.linear(X, X)
    neighbors = cos_sim.topk(1 + K)[1][:, 1:]
    neighbors = np.array(neighbors)
    return X, T, neighbors

if __name__ == '__main__':

    PLOT_IMAGES = False
    maskrcnn = MaskRCNN()

    notes_loc = 'D:/1604_notes/'
    genuine_notes_loc = 'D:/genuines/Pack_100_4/'
    dataset = 'front'
    model_directory = 'D:/models/front/golden-meadow-72_94.858/'

    args = parse_arguments()
    model = create_model(args)
    checkpoint = torch.load(model_directory + [i for i in os.listdir(model_directory) if i.endswith('.pth')][0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model_is_training = model.training
    model.eval()

    os.chdir('../data/')
    data_root = os.getcwd()

    with open(f'{model_directory}eval_coarse_dict.pkl', 'rb') as f:
        coarse_filter_dict = pickle.load(f)

    with open(f'{model_directory}eval_fine_dict.pkl', 'rb') as f:
        fine_filter_dict = pickle.load(f)

    global_csv = form_1604_frame(notes_loc)
    genuine_frame = form_genuine_frame(genuine_notes_loc)
    global_csv = pd.concat((global_csv, genuine_frame))

    notes_per_family = global_csv.groupby(['circular 1'])

    whole_note_predictions = []
    tile_predictions = []

    img_inputs = []
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]
        if pnt_key == 'NO DATA':
            continue

        valid_notes = get_valid_notes(genuine_notes_loc, notes_loc, notes_frame, ['RGB'], ['Front'])

        if len(valid_notes) > 0:
            for iter, (side, spec, pack, note_num, note_dir) in tqdm(enumerate(valid_notes),
                                                                     desc=f'{len(valid_notes)} Originals'):
                if PLOT_IMAGES:
                    fig, axs = plt.subplots(nrows=y_fac + 1, ncols=x_fac)

                root_loc = f'{notes_loc}Pack_{pack}/'
                note_image, back_note_image, seal, df = get_front_back_seal(genuine_notes_loc, maskrcnn, note_num, pack, root_loc, side, spec)
                aug_obj = augment()
                note_image = aug_obj(image=note_image)['image']
                _, tiles, y_fac, x_fac = create_tiles(note_image)

                whole_note = get_transformed_image(note_image, train=False)
                img_inputs.append(whole_note)

                if len(img_inputs) == 32:
                    batch = torch.from_numpy(np.array([np.array(tens) for tens in img_inputs]))
                    embeddings = model(batch).detach().numpy()
                    y_pred = y_pred.detach().numpy()
                    print()


                y_pred = model(whole_note[None, :])
                X, T, neighbors = add_pred_to_set(y_pred, set='eval_')
                pic = len(X) - 1
                neighbors_to_pic = neighbors[pic]
                whole_note_label = predict_label()
                whole_note_predictions.append(coarse_filter_dict[whole_note_label] == pnt_key)

                for idx, tile in enumerate(tiles):
                    row_no = idx % y_fac
                    col_no = idx // y_fac

                    im = get_transformed_image(tile)

                    y_pred = model(im[None, :])
                    X, T, neighbors = add_pred_to_set(y_pred)

                    pic = len(X) - 1
                    neighbors_to_pic = neighbors[pic]

                    y_pred_label = predict_label()
                    tile_predictions.append(coarse_filter_dict[y_pred_label] == pnt_key)

                    if PLOT_IMAGES:
                        axs[row_no, col_no].imshow(tile)
                        axs[row_no, col_no].axis('off')
                        if row_no != 0:
                            axs[row_no, col_no].set_title('PLACEHOLDER', )
                        else:
                            axs[row_no, col_no].title.set_text('PLACEHOLDER')

                        try:
                            if row_no != 0:
                                axs[row_no, col_no].set_title(coarse_filter_dict[y_pred_label], va='bottom')
                            else:
                                axs[row_no, col_no].title.set_text(coarse_filter_dict[y_pred_label])
                        except KeyError:
                            pass

                if PLOT_IMAGES:
                    axes = plt.subplot(3, 1, 3)
                    axes.imshow(note_image)
                    plt.title(f'Whole Note: Predicted: {coarse_filter_dict[whole_note_label]}')
                    plt.suptitle(f'Whole Note: Truth: {pnt_key}')
                    #plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    plt.show()
    print(len(whole_note_predictions))
    print(len(tile_predictions))
    print(sum(whole_note_predictions)/len(whole_note_predictions))
    print(sum(tile_predictions) / len(tile_predictions))


