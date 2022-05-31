import os
import pickle
import random
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


def get_X_T(set):
    X = np.load(f'{model_directory}/{set}X.npy')
    T = np.load(f'{model_directory}/{set}T.npy')
    return X, T


if __name__ == '__main__':
    PLOT_IMAGES = True
    maskrcnn = MaskRCNN()

    args = parse_arguments()

    if sys.platform != 'linux':
        notes_loc = 'D:/1604_notes/'
        genuine_notes_loc = 'D:/genuines/Pack_100_4/'
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
            print([LOG_DIR + i for i in os.listdir(LOG_DIR) if os.path.isdir(i)])
            model_directory = sorted([LOG_DIR + i for i in os.listdir(LOG_DIR) if os.path.isdir(i)], key= lambda i: int(i.split('_')[-1]))[0]
            print(f'model directory is {model_directory}')
        else:
            print(f'{LOG_DIR} does not exist')
            sys.exit()

    dataset = 'front'

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

    global_csv = form_1604_frame(notes_loc)
    genuine_frame = form_genuine_frame(genuine_notes_loc)
    global_csv = pd.concat((global_csv, genuine_frame))

    notes_per_family = global_csv.groupby(['circular 1'])

    whole_note_predictions = []
    tile_predictions = []

    X_dont_change, T_dont_change = get_X_T('eval_')
    X_dont_change_val, T_dont_change_val = get_X_T('val_')

    img_inputs = []
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]
        if pnt_key == 'NO DATA':
            continue

        valid_notes = get_valid_notes(genuine_notes_loc, notes_loc, notes_frame, ['RGB'], ['Front'])

        if len(valid_notes) > 0:
            for iter, (side, spec, pack, note_num, note_dir) in tqdm(enumerate(valid_notes),
                                                                     desc=f'{len(valid_notes)} Originals'):
                root_loc = f'{notes_loc}Pack_{pack}/'
                note_image, back_note_image, seal, df = get_front_back_seal(note_dir, maskrcnn)
                aug_obj = augment()
                note_image = aug_obj(image=note_image)['image']
                _, tiles, y_fac, x_fac = create_tiles(note_image)

                whole_note = get_transformed_image(note_image, train=False)
                y_pred = model(whole_note[None, :])

                Xhat, That, neighbors = add_pred_to_set(y_pred, X_dont_change, T_dont_change)
                pic = len(Xhat) - 1
                neighbors_to_pic = neighbors[pic]
                y_pred_label = predict_label(Xhat, That, neighbors_to_pic, pic)

                whole_note_label = coarse_filter_dict_saved[y_pred_label]
                whole_note_predictions.append(whole_note_label == pnt_key)

                if PLOT_IMAGES:
                    fig, axs = plt.subplots(nrows=y_fac + 1, ncols=x_fac)

                for idx, tile in enumerate(tiles):
                    row_no = idx % y_fac
                    col_no = idx // y_fac

                    im = get_transformed_image(tile)

                    y_pred = model(im[None, :])
                    Xhat, That, neighbors = add_pred_to_set(y_pred, X_dont_change_val, T_dont_change_val)

                    pic = len(Xhat) - 1
                    neighbors_to_pic = neighbors[pic]

                    y_pred_label = predict_label(Xhat, That, neighbors_to_pic, pic)
                    tile_predictions.append(coarse_filter_dict_saved_val[y_pred_label] == pnt_key)

                    if PLOT_IMAGES:
                        axs[row_no, col_no].imshow(tile)
                        axs[row_no, col_no].axis('off')
                        if row_no != 0:
                            axs[row_no, col_no].set_title('PLACEHOLDER', )
                        else:
                            axs[row_no, col_no].title.set_text('PLACEHOLDER')

                        try:
                            if row_no != 0:
                                axs[row_no, col_no].set_title(coarse_filter_dict_saved[y_pred_label], va='bottom')
                            else:
                                axs[row_no, col_no].title.set_text(coarse_filter_dict_saved[y_pred_label])
                        except KeyError:
                            pass

                if PLOT_IMAGES:
                    axes = plt.subplot(3, 1, 3)
                    axes.imshow(note_image)
                    plt.title(f'Whole Note: Predicted: {whole_note_label}')
                    plt.suptitle(f'Whole Note: Truth: {pnt_key}')
                    #plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05, hspace=0.25)
                    os.makedirs(f'../training/{args.dataset}/{model_directory.split("/")[-2]}/plots/', exist_ok=True)
                    plt.savefig(f'../training/{args.dataset}/{model_directory.split("/")[-2]}/plots/{os.path.splitext(os.path.split(note_dir)[-1])}.png')
                    plt.close()
    print(len(whole_note_predictions))
    print(len(tile_predictions))
    print(sum(whole_note_predictions)/len(whole_note_predictions))
    print(sum(tile_predictions) / len(tile_predictions))


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