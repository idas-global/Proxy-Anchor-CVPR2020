import os
import functools
import pandas as pd
import numpy as np
import torch
import random
from matplotlib.colors import ListedColormap
import networkx as nx

from scipy.spatial import distance_matrix
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from tqdm import tqdm
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()
import mplcursors


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

    predictions = np.zeros((num_batches, batch_sz, 512))
    labels = np.zeros((num_batches, batch_sz))

    if return_images:
        # TODO add a attribute to dataloader that gives the shape
        #image_array = np.zeros((len(dataloader.dataset.im_paths), 3, 224, 224))
        image_array = np.zeros((num_batches, batch_sz, 3, 224, 224))

    missing_batch = 0

    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc='Getting Predictions'):
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

    model.train()
    model.train(model_is_training) # revert to previous training state

    if return_images:
        image_array = combine_dims(image_array, 0, 1)
        image_array = image_array[0:-missing_batch, :]

    predictions = combine_dims(predictions, 0, 1)
    predictions = predictions[0:-missing_batch, :]

    labels = combine_dims(labels, 0, 1)
    labels = labels[0:-missing_batch]

    if return_images:
        return [torch.from_numpy(predictions), torch.from_numpy(labels), torch.from_numpy(image_array)]
    return [torch.from_numpy(predictions), torch.from_numpy(labels)]


def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean


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


def evaluate_cos(model, dataloader, epoch, args):
    # calculate embeddings with model and get targets
    dest = f'../training/{args.dataset}/{epoch}/'
    os.makedirs(dest, exist_ok=True)

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

    recall = {}

    if epoch % 2 == 0:
        coarse_filter_dict, fine_filter_dict, metrics = get_accuracies(T, X, dataloader, neighbors)
        recall['specific_accuracy'] = metrics['specific_accuracy']
        recall['coarse_accuracy'] = metrics['coarse_accuracy']
    #plot_feature_space(X, dataloader)

    for k in [1, 2, 4, 8, 16, 32]:
        y_preds = []
        for t, y in zip(T, Y):
            y_preds.append(torch.mode(torch.Tensor(y).long()[:k]).values)

        y_preds = np.array(y_preds).astype(int)
        y_true = np.array(T)
        r_at_k = f1_score(y_true, y_preds, average='weighted')
        recall[f"f1score@{k}"] = r_at_k
        print("f1score@{} : {:.3f}".format(k, 100 * r_at_k))

        if epoch % 2 == 0:
            metrics[f'f1score@{k}'] = r_at_k*100

    if epoch % 2 == 0:
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

        params = ['prediction', 'truth']
        degrees = ['fine', 'coarse']
        for deg in degrees:
            for para in params:
                centroids = plot_node_graph(X, data_viz_frame, para, deg, dest)
        #
        # distance = []
        # for idx, bird in data_viz_frame.iterrows():
        #     distance.append(sum(abs(X[idx] - centroids[bird['truth_label_coarse']])))
        #
        # data_viz_frame['distance_from_center_coarse'] = np.array(distance)
        # ax = sns.violinplot(x="truth_label_coarse", y="distance_from_center_coarse",
        #                     data=data_viz_frame, palette="muted")
        #
        # fig = plt.Figure(figsize=(12,12))
        # #ax = plt.axes(projection='3d')
        # ax = plt.axes()
        # pca = KernelPCA(n_components=2, kernel='linear')
        # new_x = pca.fit_transform(X)
        #
        # for idx, (coarse, frame) in enumerate(data_viz_frame.groupby('truth_label_coarse')):
        #     ax.scatter(new_x[frame.index][:,0],
        #                new_x[frame.index][:,1],
        #                #new_x[frame.index][:,2],
        #                label=coarse, s=40)
        # mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        # ax.legend()
        # plt.show(block=True)

        # f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
        # f.suptitle('Bird Dataset Metric Learning', fontsize=14)
        #
        # sns.boxplot(x="Birds", y="alcohol", data=wines, ax=ax)
        # ax.set_xlabel("Wine Quality", size=12, alpha=0.8)
        # ax.set_ylabel("Wine Alcohol %", size=12, alpha=0.8)

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
                        annot=annot, fmt='g', ax=ax, annot_kws={"size" : 4}) # annot=True to annotate cells, ftm='g' to disable scientific notation

            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Confusion Matrix for {param}')
            ax.xaxis.set_ticklabels(np.unique(data_viz_frame[f'prediction_{param}'].values))
            ax.yaxis.set_ticklabels(np.unique(data_viz_frame[f'truth_{param}'].values))
            plt.savefig(dest + f'Confusion_{param}.png', bbox_inches='tight', dpi=300)
            plt.close()

        metrics.to_csv(dest + 'metrics.csv')
    return recall


def plot_node_graph(X, data_viz_frame, para, deg, dest):
    centroids = {}
    for idx, (coarse, frame) in enumerate(data_viz_frame.groupby(f'{para}_label_{deg}')):
        centroids[coarse] = X[frame.index].mean(axis=0)
    centroids_af = pd.DataFrame(centroids).T

    dst_matrix = distance_matrix(centroids_af, centroids_af, p=0.1)

    fig = plt.Figure(figsize=(120, 60))

    G = nx.from_numpy_matrix(dst_matrix)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), centroids_af.index)))
    import itertools
    edges = nx.minimum_spanning_edges(G)
    weights = [edge[2]['weight'] for edge in edges]
    weights = np.array(weights) * 20 / max(weights)
    weights = [round(x, 2) for x in weights]

    G2 = nx.create_empty_copy(G)
    edges = nx.minimum_spanning_edges(G)
    for edge, wgt in zip(edges, weights):
        edge = (edge[0], edge[1])
        G2.add_edge(*edge, weight=wgt)

    pos = nx.kamada_kawai_layout(G2)
    nx.draw_networkx_labels(G2, pos=pos, bbox=dict(
        boxstyle='round', ec=(0.0, 0.0, 0.0),
        alpha=0.9, fc='white', lw=1.5
    ),
                            verticalalignment='center', ax=fig.add_subplot(111))
    nx.draw(G2, node_size=600, pos=pos, with_labels=False, ax=fig.add_subplot(111))
    plt.show(block=False)
    fig.savefig(f'{dest}/node_graph_{para}_{deg}.png', bbox_inches='tight', dpi=200)
    return centroids


def get_accuracies(T, X, dataloader, neighbors):
    pictures_to_predict = random.choices(range(len(X)), k=int(round(len(dataloader.dataset.im_paths)*1/100)))
    ground_truth = T[pictures_to_predict]

    coarse_filter_dict = {class_num: specific_species
                          for class_num, specific_species in zip(np.array(T), dataloader.dataset.class_names_coarse)}

    fine_filter_dict = {class_num : specific_species
                        for class_num, specific_species in zip(np.array(T), dataloader.dataset.class_names_fine)}

    y_preds = np.zeros(len(pictures_to_predict))
    for idx, pic in tqdm(enumerate(pictures_to_predict), total=len(pictures_to_predict), desc='Accuracy Analysis'):
        neighbors_to_pic = [neigh.item() for neigh in neighbors[pic] if neigh not in pictures_to_predict]
        y_preds[idx] = torch.mode(T[neighbors_to_pic].long()).values

    print(f'Accuracy at Specfic: {accuracy_score(y_preds, np.array(ground_truth)) * 100}')

    coarse_predictions = [coarse_filter_dict[pred] for pred in y_preds]
    coarse_truth = [coarse_filter_dict[truth] for truth in np.array(ground_truth)]
    print(f'Accuracy at Coarse: {accuracy_score(coarse_predictions, coarse_truth) * 100}')

    metrics = pd.DataFrame([accuracy_score(y_preds, np.array(ground_truth)) * 100])
    metrics.columns = ['specific_accuracy']
    metrics['coarse_accuracy'] = accuracy_score(coarse_predictions, coarse_truth) * 100
    return coarse_filter_dict, fine_filter_dict, metrics

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
