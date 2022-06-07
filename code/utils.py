import os
import functools
import traceback
import uuid
import itertools
import warnings
from collections import Counter
from operator import itemgetter
import pandas as pd
import numpy as np
import torch
import random
from matplotlib.colors import ListedColormap
import networkx as nx

from scipy.spatial import distance_matrix
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

# from dsprofiling.src.clustering_tools import break_down_clusters

plt.ioff()
import mplcursors


def l2_norm(input):
    buffer = input ** 2
    normp = np.sqrt(np.sum(buffer, axis=1))
    output = input / normp[:, None]
    return output


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t == torch.mode(torch.Tensor(y).long()[:k]).values:
            s += 1
    return s / (1. * len(T))


def combine_dims(a, i=0, n=1):
    """
  Combines dimensions of numpy array `a`,
  starting at index `i`,
  and combining `n` dimensions
  """
    s = list(a.shape)
    combined = functools.reduce(lambda x, y: x * y, s[i:i + n + 1])
    return np.reshape(a, s[:i] + [combined] + s[i + n + 1:])


def predict_batchwise(model, dataloader, return_images=False):
    batch_sz = dataloader.batch_sampler.batch_size
    num_batches = int(np.ceil(len(dataloader.dataset.ys) / batch_sz))

    predictions = np.zeros((num_batches, batch_sz, 512))
    labels = np.zeros((num_batches, batch_sz))

    if return_images:
        # TODO add a attribute to dataloader that gives the shape
        # image_array = np.zeros((len(dataloader.dataset.im_paths), 3, 224, 224))
        image_array = np.zeros((num_batches, batch_sz, 3, 448, 448))

    missing_batch = 0

    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for idx in tqdm(range(num_batches), desc='Getting Predictions'):
            image = []
            target = []
            for i in range(batch_sz):
                try:
                    x, y = dataloader.dataset.__getitem__(i + idx*batch_sz)
                except IndexError:
                    break
                image.append(x)
                target.append(y)
            image = torch.stack(image)
            target = torch.Tensor(target).int()
            batch = [image, target]
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if len(J) != batch_sz:
                    filled_batch = np.array(J, dtype=np.float32)
                    empty_batch = np.zeros((batch_sz - len(J), *filled_batch.shape[1::]), dtype=np.float32)
                    missing_batch = len(empty_batch)

                    if i == 1:
                        J = torch.from_numpy(np.hstack((filled_batch, empty_batch)))
                    else:
                        J = torch.from_numpy(np.vstack((filled_batch, empty_batch)))

                if i == 0:
                    if return_images:
                        image_array[i, :] = J

                    # move images to device of model (approximate device)
                    J = model(J)
                    predictions[idx, :] = J
                else:
                    labels[idx, :] = J

    if not missing_batch:
        missing_batch = -1 * num_batches * batch_sz

    if return_images:
        image_array = combine_dims(image_array, 0, 1)
        image_array = image_array[0:-missing_batch, :]

    predictions = combine_dims(predictions, 0, 1)
    predictions = predictions[0:-missing_batch, :]

    labels = combine_dims(labels, 0, 1)
    labels = labels[0:-missing_batch]

    if return_images:
        return [predictions, labels, image_array]
    return [predictions, labels]


def parse_im_name(specific_species, exclude_trailing_consonants=False, fine=False):
    if fine:
        filter = os.path.split(os.path.split(specific_species)[0])[1].split('.')[-1].lower()
    else:
        coarse_filter = os.path.split(os.path.split(specific_species)[0])[1].split('_')[-1].lower()
        if '.' in coarse_filter:
            coarse_filter = coarse_filter.split('.')[-1]
        filter = coarse_filter

    if exclude_trailing_consonants:
        if filter[-1].isalpha():
            filter = filter[0:-1]
    return filter


def get_X_T_Y(dataloader, model, validation):
    X, T = predict_batchwise(model, dataloader, return_images=False)

    if validation is not None:
        X2, T2 = predict_batchwise(model, validation, return_images=False)
        X = np.vstack((X, X2))
        T = np.hstack((T, T2))

    T = torch.from_numpy(T)
    X = l2_norm(X)
    X = torch.from_numpy(X)
    # get predictions by assigning nearest 8 neighbors with cosine

    K = min(64, len(X) - 1)
    cos_sim = F.linear(X, X)
    neighbors = cos_sim.topk(1 + K)[1][:, 1:]

    Y = T[neighbors]
    Y = Y.float().cpu()

    neighbors = neighbors.numpy()
    return X, T, Y, neighbors


def save_metrics(dataloader, metrics, train_dest, val_dest, test_dest):
    if dataloader.dataset.mode == 'train':
        metrics.to_csv(train_dest + 'metrics.csv')
    if dataloader.dataset.mode == 'validation':
        metrics.to_csv(val_dest + 'metrics.csv')
    if dataloader.dataset.mode == 'eval':
        metrics.to_csv(test_dest + 'metrics.csv')

def confusion_matrices(data_viz_frame, dataloader, train_dest, val_dest, test_dest):
    if dataloader.dataset.mode == 'train':
        plot_confusion(data_viz_frame, dataloader, train_dest)
    if dataloader.dataset.mode == 'validation':
        plot_confusion(data_viz_frame, dataloader, val_dest)
    if dataloader.dataset.mode == 'eval':
        plot_confusion(data_viz_frame, dataloader, test_dest)

def create_and_save_viz_frame(X, dataloader, coarse_filter_dict, fine_filter_dict,
                              pictures_to_predict,
                              train_dest, val_dest, test_dest,
                              y_preds, y_true):
    data_viz_frame = form_data_viz_frame(X[pictures_to_predict], coarse_filter_dict, fine_filter_dict,
                                         dataloader, y_preds, y_true)

    if dataloader.dataset.mode == 'train':
        data_viz_frame.to_csv(train_dest + 'train_and_validation_data_combined.csv')
    if dataloader.dataset.mode == 'validation':
        data_viz_frame.to_csv(val_dest + 'validation_data.csv')
    if dataloader.dataset.mode == 'eval':
        data_viz_frame.to_csv(test_dest + 'test_data.csv')
    return data_viz_frame


def plot_relationships(X, data_viz_frame, dataloader, deg, para, pictures_to_predict,
                       train_dest, val_dest, test_dest):
    if dataloader.dataset.mode == 'train':
        plot_node_graph(X[pictures_to_predict], data_viz_frame, para, deg, train_dest)
    if dataloader.dataset.mode == 'validation':
        plot_node_graph(X[pictures_to_predict], data_viz_frame, para, deg, val_dest)
    if dataloader.dataset.mode == 'eval':
        plot_node_graph(X[pictures_to_predict], data_viz_frame, para, deg, test_dest)



def calc_recall(T, Y, k, prepend=''):
    metrics = pd.DataFrame()
    y_preds = []
    for t, y in zip(T, Y):
        y_preds.append(torch.mode(torch.Tensor(y).long()[:k]).values)
    y_preds = np.array(y_preds).astype(int)
    y_true = np.array(T)
    r_at_k = f1_score(y_true, y_preds, average='weighted')
    metrics[f'{prepend}f1score@{k}'] = [r_at_k * 100]
    return y_preds, y_true, metrics


def plot_confusion(data_viz_frame, dataloader, dest):
    params = ['label_coarse', 'label_fine', 'denom']
    if 'Rupert_Book' not in dataloader.dataset.im_paths[0]:
        params = ['label_coarse', 'label_fine']

    for param in params:
        fig = plt.Figure(figsize=(48, 48))
        ax = plt.subplot()
        annot = True
        if param == 'label_fine':
            annot = False
        sns.heatmap(confusion_matrix(data_viz_frame[f'prediction_{param}'].values,
                                     data_viz_frame[f'truth_{param}'].values),
                    annot=annot, fmt='g', ax=ax,
                    annot_kws={"size": 4})  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix for {param}')
        ax.xaxis.set_ticklabels(np.unique(data_viz_frame[f'prediction_{param}'].values))
        ax.yaxis.set_ticklabels(np.unique(data_viz_frame[f'truth_{param}'].values))
        plt.savefig(dest + f'Confusion_{param}.png', bbox_inches='tight', dpi=300)
        plt.close()


def form_data_viz_frame(X, coarse_filter_dict, fine_filter_dict, dataloader, y_preds, y_true):
    data_viz_frame = pd.DataFrame(y_true.astype(int), columns=['truth'])
    data_viz_frame['prediction'] = y_preds
    data_viz_frame['truth_label_coarse'] = data_viz_frame['truth'].map(coarse_filter_dict, y_true)
    data_viz_frame['prediction_label_coarse'] = data_viz_frame['prediction'].map(coarse_filter_dict, y_preds)
    data_viz_frame['truth_label_fine'] = data_viz_frame['truth'].map(fine_filter_dict, y_true)
    data_viz_frame['prediction_label_fine'] = data_viz_frame['prediction'].map(fine_filter_dict, y_preds)
    if 'Rupert_Book' in dataloader.dataset.im_paths[0]:
        data_viz_frame['truth_denom'] = [label.split('_')[0] for label in data_viz_frame['truth_label_fine']]
        data_viz_frame['prediction_denom'] = [label.split('_')[0] for label in data_viz_frame['prediction_label_fine']]
    data_viz_frame['mean_coarse'] = X.mean(axis=1)
    return data_viz_frame


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def plot_node_graph(X, data_viz_frame, para, deg, dest):
    # TODO make this works for individual points
    centroids = {}
    full_items = []
    for idx, (coarse, frame) in enumerate(data_viz_frame.groupby(f'{para}_label_{deg}')):
        centroids[coarse] = X[frame.index].mean(axis=0)

        for note in frame.index:
            full_items.append((coarse, *X[note]))

    centroids_af = pd.DataFrame(centroids).T
    dst_matrix = distance_matrix(centroids_af, centroids_af, p=0.1)

    sca = MinMaxScaler()
    dst_matrix = sca.fit_transform(dst_matrix)
    fig = plt.Figure(figsize=(120, 60))

    G = nx.from_numpy_matrix(dst_matrix)

    weight_dict = {}
    for (i, j) in itertools.permutations(range(len(G.nodes)), 2):
        if i not in weight_dict.keys():
            weight_dict[i] = {}
        weight_dict[i][j] = G.get_edge_data(i, j)['weight']

    G = nx.create_empty_copy(G)

    num_nbors = 1
    for i in range(len(G.nodes)):

        res = dict(sorted(weight_dict[i].items(), key=itemgetter(1))[:num_nbors])
        small_edges = [key for key, val in res.items() if val < 0.6]

        for edg in small_edges:
            if i == edg:
                print(small_edges)
                print(i)
            G.add_edge(i, edg, weight=weight_dict[i][edg])

    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),
                                     [i for i in centroids_af.index])))

    pos = nx.spring_layout(G)

    nx.draw_networkx_labels(G, pos=pos, bbox=dict(
        boxstyle='round', ec=(0.0, 0.0, 0.0),
        alpha=0.9, fc='white', lw=1.5
    ),
                            verticalalignment='center', ax=fig.add_subplot(111))
    nx.draw(G, node_size=600, pos=pos, with_labels=False, ax=fig.add_subplot(111))
    plt.show(block=False)
    fig.savefig(f'{dest}{para}_{deg}_graph.png', bbox_inches='tight', dpi=200)
    return centroids


import math


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def f1_score_calc(T, Y, dataloader, pictures_to_predict):
    for k in [3, 5, 7]:
        val_y_preds, val_y_true, metrics = calc_recall(T[pictures_to_predict], Y[pictures_to_predict],
                                                       k, prepend=dataloader.dataset.mode + '_')
    return metrics


def get_accuracies(T, X, dataloader, neighbors, pictures_to_predict, metrics):
    ground_truth = T[pictures_to_predict]

    coarse_filter_dict = dataloader.dataset.class_names_coarse_dict

    fine_filter_dict = dataloader.dataset.class_names_fine_dict

    y_preds = np.zeros(len(pictures_to_predict))
    y_preds_mode = np.zeros(len(pictures_to_predict))
    for idx, pic in tqdm(enumerate(pictures_to_predict), total=len(pictures_to_predict), desc='Accuracy Analysis'):
        neighbors_to_pic = np.array(neighbors[pic, :][~np.in1d(neighbors[pic, :], pictures_to_predict)])

        preds, counts = np.unique(T[neighbors_to_pic[0:7]], return_counts=True)
        if len(counts) == 0:
            y_preds[idx] = 9999
            y_preds_mode[idx] = 9999
            continue

        try:
            close_preds = preds[counts >= np.max(counts) - 1]
        except Exception:
            print(counts)
            print(traceback.format_exc())
            y_preds[idx] = 9999
            y_preds_mode[idx] = 9999
            continue
            
        y_preds_mode[idx] = preds[np.argsort(counts)[-1]]
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

        y_preds[idx] = max(predictions, key=predictions.get)

    ground_truth = np.array(ground_truth)
    coarse_predictions = [coarse_filter_dict[pred] for pred in y_preds]
    coarse_truth = [coarse_filter_dict[truth] for truth in ground_truth]

    metrics[f'{dataloader.dataset.mode}_specific_accuracy'] = accuracy_score(y_preds, ground_truth) * 100
    metrics[f'{dataloader.dataset.mode}_specific_mode_accuracy'] = accuracy_score(y_preds_mode, ground_truth) * 100
    metrics[f'{dataloader.dataset.mode}_coarse_accuracy'] = accuracy_score(coarse_predictions, coarse_truth) * 100

    tally_true = dict(Counter(y_preds))
    for key, val in sorted(dict(Counter(y_preds)).items()):
        pred_fac = np.round(100 * val / len(pictures_to_predict), 2)
        true_fac = np.round(100 * tally_true[key] / len(pictures_to_predict), 2)
        if abs(pred_fac - true_fac) > 3:
            print(f'{int(key)}: pred: {val} - {pred_fac}% vs true: {true_fac}%')

    return coarse_filter_dict, fine_filter_dict, y_preds, ground_truth
