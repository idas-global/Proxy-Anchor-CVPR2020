import pickle
import sys

import mplcursors
import matplotlib.pyplot as plt
import numpy as np
import os

from apply_model_across_note import plot_and_return_fig
from utils import parse_arguments


def main(list_of_embed_paths):
    embed_path = f'/{args.dataset}'
    embeddings = np.zeros((0, 512))
    circ_labels = np.zeros((0,))
    note_labels = np.zeros((0,))

    for path in list_of_embed_paths:
        embeddings_existing = np.load(path + f'{embed_path}_embeddings.npy', allow_pickle=True)
        circ_labels_existing = np.load(path + f'{embed_path}_circ_labels.npy', allow_pickle=True)
        note_labels_existing = np.load(path + f'{embed_path}_note_labels.npy', allow_pickle=True)
        # fig = plot_and_return_fig(embeddings_existing, circ_labels_existing, note_labels_existing)

        #trans = umap.UMAP(n_neighbors=5, random_state=42).fit(embeddings_existing)

        print('COLLATING EMBEDDINGS')

        embeddings = np.vstack((embeddings, embeddings_existing))
        circ_labels = np.hstack((circ_labels, circ_labels_existing))
        note_labels = np.hstack((note_labels, note_labels_existing))

    _, unique_idxs = np.unique(note_labels, return_index=True)

    embeddings = embeddings[unique_idxs, :]
    circ_labels = circ_labels[unique_idxs]
    note_labels = note_labels[unique_idxs]

    os.makedirs(outpath, exist_ok=True)
    if os.path.isfile(outpath + f'{embed_path}_embeddings.npy'):
        sys.exit('Embeddings already Exist, change outpath')

    np.save(outpath + f'{embed_path}_embeddings.npy', embeddings)
    np.save(outpath + f'{embed_path}_circ_labels.npy', circ_labels)
    np.save(outpath + f'{embed_path}_note_labels.npy', note_labels)
    return embeddings, circ_labels, note_labels

if __name__ == '__main__':
    args = parse_arguments()
    list_of_embed_paths = ['D:/model_outputs/proxy_anchor/training/note_families_front/ancient-brook-119/true_validation/acceptable maybe/',
                           'D:/model_outputs/proxy_anchor/training/note_families_front/babbling-tree-333/true_validation_-1/']

    list_of_embed_paths = [list_of_embed_paths[0]]
    outpath = 'D:/model_outputs/proxy_anchor/training/note_families_front/ancient-brook-119/collated_true_validation/'
    embeddings, circ_labels, note_labels = main(list_of_embed_paths)
    fig = plot_and_return_fig(embeddings, circ_labels, note_labels)
    pickle.dump(fig, open(outpath + 'nonaug_truth_fine_tSNE.pkl', 'wb'))

