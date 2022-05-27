import os
import sys

import PIL
import cv2
import numpy as np
import torch

from utils import l2_norm
from dataset.utils import RGBToBGR, ScaleIntensities
from noteclasses import ImageBMP
from train import parse_arguments, create_model, create_generators
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd


def transform(im):
    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
        [
            RGBToBGR(),
            transforms.Resize((inception_sz_resize, inception_sz_resize * 2)),
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


def create_tiles():
    note_dir = os.path.join(root, file)
    note = ImageBMP(note_dir, straighten=True, rotation=None)
    im = note.array

    tiles = []
    pictures = 3
    edges = np.linspace(0, im.shape[1], pictures + 1)
    for start, end in zip(edges[:-1], edges[1:]):
        box = (start, 0, end, im.shape[0]/2)
        tiles.append(im[int(round(box[1])):int(round(box[3])), int(round(box[0])):int(round(box[2]))])
        box = (start, im.shape[0] / 2, end, im.shape[0])
        tiles.append(im[int(round(box[1])):int(round(box[3])), int(round(box[0])):int(round(box[2]))])

    return note_dir, tiles, 2, 3


def get_transformed_image():
    tile_dest = f'./note_tiles/{note_dir.split("/")[-1]}'[0:-4] + f'_tile_{idx}.bmp'
    os.makedirs(os.path.split(tile_dest)[0], exist_ok=True)
    cv2.imwrite(tile_dest, tile)
    im = PIL.Image.open(tile_dest)
    # convert gray to rgb
    if len(list(im.split())) == 1: im = im.convert('RGB')
    im = transform(im)
    return im


def add_pred_to_set():
    X = np.load(os.path.split(model_directory)[0] + '/X.npy')
    X = np.vstack((X, y_pred.detach().numpy()))
    X = l2_norm(X)
    X = torch.from_numpy(X)

    T = np.load(os.path.split(model_directory)[0] + '/T.npy')
    T = np.hstack((T, -1))
    T = torch.from_numpy(T)

    K = min(32, len(X) - 1)

    cos_sim = F.linear(X, X)
    neighbors = cos_sim.topk(1 + K)[1][:, 1:]
    neighbors = np.array(neighbors)
    return X, T, neighbors

if __name__ == '__main__':

    notes_loc = 'D:/1604_notes/'
    dataset = 'front'
    model_directory = 'D:/models/front/feasible-water-68_82.599/'

    args = parse_arguments()

    model = create_model(args)
    model_is_training = model.training
    model.eval()

    os.chdir('../data/')
    data_root = os.getcwd()

    checkpoint = torch.load(model_directory + os.listdir(model_directory)[0])
    model.load_state_dict(checkpoint['model_state_dict'])

    ds = 'train_and_validation'
    epoch = 29
    validation_data = pd.read_csv(os.path.split(os.path.split(model_directory)[0])[0] + f'/{ds}/{ds}_data_combined.csv', index_col=0)
    fine_filter_dict = dict(zip(validation_data['prediction'], validation_data['prediction_label_fine']))
    fine_filter_dict = dict(sorted(fine_filter_dict.items()))

    coarse_filter_dict = dict(zip(validation_data['prediction'], validation_data['prediction_label_coarse']))
    coarse_filter_dict = dict(sorted(coarse_filter_dict.items()))

    for (root, dirs, files) in os.walk(notes_loc):
        for file in files:
            if 'RGB' in file and '.bmp' in file and dataset in file.lower():
                note_dir, tiles, y_fac, x_fac = create_tiles()

                fig, axs = plt.subplots(nrows=y_fac, ncols=x_fac)

                for idx, tile in enumerate(tiles):
                    row_no = idx % y_fac
                    col_no = idx // y_fac

                    im = get_transformed_image()

                    y_pred = model(im[None, :])
                    X, T, neighbors = add_pred_to_set()

                    pic = len(X) - 1
                    neighbors_to_pic = neighbors[pic]

                    y_pred_label = predict_label()
                    axs[row_no, col_no].imshow(tile)
                    axs[row_no, col_no].title.set_text('PLACEHOLDER')
                    axs[row_no, col_no].axis('off')

                    try:
                        axs[row_no, col_no].title.set_text(coarse_filter_dict[y_pred_label])
                    except KeyError:
                        pass
                plt.show()



