import os
import functools

import mplcursors
import numpy as np
import torch
import logging
import random
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
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
  combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
  return np.reshape(a, s[:i] + [combined] + s[i+n+1:])


def predict_batchwise(model, dataloader, return_images=False):
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset

    batch_sz = dataloader.batch_sampler.batch_size
    num_batches = len(dataloader.batch_sampler)

    predictions = np.zeros((len(dataloader.dataset.im_paths), 512))
    labels = np.zeros(len(dataloader.dataset.im_paths))

    if return_images:
        # TODO add a attribute to dataloader that gives the shape
        #image_array = np.zeros((len(dataloader.dataset.im_paths), 3, 224, 224))
        image_array = np.zeros((num_batches, batch_sz, 3, 224, 224))

    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc='Getting Predictions'):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices

                if i == 0:
                    if return_images:
                        image_array[i, :] = J
    image_array = combine_dims(image_array, 0, 1)
    print(image_array.shape)
    print(len(dataloader.dataset.im_paths))
    #                 # move images to device of model (approximate device)
    #                 J = model(J)
    #                 predictions[(idx*batch_sz):((idx+1)*batch_sz)] = J
    #             else:
    #                 labels[(idx * batch_sz):((idx + 1) * batch_sz)] = J
    #
    # model.train()
    # model.train(model_is_training) # revert to previous training state
    #
    # if return_images:
    #     return [torch.from_numpy(predictions), torch.from_numpy(labels), torch.from_numpy(image_array)]
    # return [torch.from_numpy(predictions), torch.from_numpy(labels)]


def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean


def parse_im_name(specific_species, exclude_trailing_consonants=False):
    coarse_filter = os.path.split(os.path.split(specific_species)[0])[1].split('_')[-1].lower()
    if '.' in coarse_filter:
        coarse_filter = coarse_filter.split('.')[-1]

    if exclude_trailing_consonants:
        if coarse_filter[-1].isalpha():
            coarse_filter = coarse_filter[0:-1]
    return coarse_filter


def evaluate_cos(model, dataloader, epoch):
    # calculate embeddings with model and get targets
    if epoch % 2 == 0:
        X, T, I = predict_batchwise(model, dataloader, return_images=True)
    else:
        X, T = predict_batchwise(model, dataloader, return_images=False)

    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    neighbors = cos_sim.topk(1 + K)[1][:, 1:]
    Y = T[neighbors]
    Y = Y.float().cpu()

    if epoch % 2 == 0:
        get_accuracies(T, X, dataloader, neighbors)

    #plot_feature_space(X, dataloader)

    recall = []
    for k in [32]:
        y_preds = []
        for t, y in zip(T, Y):
            y_preds.append(torch.mode(torch.Tensor(y).long()[:k]).values)

        y_preds = np.array(y_preds).astype(int)
        y_true = np.array(T)
        r_at_k = f1_score(y_true, y_preds, average='weighted')
        recall.append(r_at_k)
        print("f1score@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall


def get_accuracies(T, X, dataloader, neighbors):
    pictures_to_predict = random.choices(range(len(X)), k=2000)
    ground_truth = T[pictures_to_predict]
    coarse_filter_dict = {class_num: parse_im_name(specific_species, exclude_trailing_consanants='Rupert_Book_Augmented' in dataloader.dataset.root)
                          for class_num, specific_species in zip(np.array(T), dataloader.dataset.im_paths)}
    y_preds = []
    for pic in tqdm(pictures_to_predict, desc='Accuracy Analysis'):
        neighbors_to_pic = [neigh.item() for neigh in neighbors[pic] if neigh not in pictures_to_predict]
        y_preds.append(torch.mode(T[neighbors_to_pic].long()).values)
    print(f'Accuracy at Specfic: {accuracy_score(np.array(y_preds), np.array(ground_truth)) * 100}')
    coarse_predictions = [coarse_filter_dict[pred] for pred in np.array(y_preds)]
    coarse_truth = [coarse_filter_dict[truth] for truth in np.array(ground_truth)]
    print(f'Accuracy at Coarse: {accuracy_score(coarse_predictions, coarse_truth) * 100}')


def plot_feature_space(X, dataloader):
    inspect_space = np.array(X)
    kernel_key = 'sigmoid'
    pca = KernelPCA(n_components=3, kernel=kernel_key)
    transformed_space = pca.fit_transform(inspect_space)
    coarse_filters = [parse_im_name(specific_species) for specific_species in dataloader.dataset.im_paths]
    plot_based_on_cluster_id(np.array(coarse_filters), {kernel_key: transformed_space}, kernel=kernel_key)


def plot_based_on_cluster_id(cluster_id, transformedDfs, kernel='all'):
    cmap = ListedColormap(sns.color_palette("husl", len(np.unique(cluster_id))).as_hex())
    colours = {pnt: cmap.colors[idx] for idx, pnt in enumerate(np.unique(cluster_id))}

    for kernel_key, transformedDf in transformedDfs.items():
        if kernel != 'all' and kernel != kernel_key:
            continue

        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        for pnt in np.unique(cluster_id):
            pnt_bool = [pnt == ii for ii in cluster_id]
            ax.scatter(transformedDf[pnt_bool, 0],
                       transformedDf[pnt_bool, 1],
                       transformedDf[pnt_bool, 2],
                       s=40, c=colours[pnt], marker='o', alpha=1, label=pnt)
        ax.legend()
        mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        # save
        fig.suptitle(kernel_key)
        plt.show()


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall
