import cv2
import cmapy
import random
import numpy as np
import open3d as o3d
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from utils import l2_norm


def to_uint8(arr):
    return (255*(arr - arr.min()) / arr.max()).astype(np.uint8)


def mapLabelsToColours(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    #return cv2.applyColorMap(to_uint8(labels), cmapy.cmap('husl')) / 255.0
    cmap = ListedColormap(sns.color_palette("husl", len(np.unique(labels))))
    colours = {pnt: cmap.colors[idx] for idx, pnt in enumerate(np.unique(labels))}
    col = [colours[i] for i in labels]
    return np.array(col)



def mapDataAsPointCloud(data, data_other):
    pts    = data[:, :3]
    labels = data[:, 3]
    pts_other    = data_other[:, :3]
    labels_other = data_other[:, 3]
    colors = mapLabelsToColours(labels)
    colors_other = mapLabelsToColours(labels_other)

    pcd = o3d.geometry.PointCloud()
    pcd_other = o3d.geometry.PointCloud()

    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_other.colors = o3d.utility.Vector3dVector(colors_other)

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_other.points = o3d.utility.Vector3dVector(pts_other)

    o3d.visualization.draw([pcd, pcd_other], show_ui=True)


def createRandomClusterData(n=1200):
    data = np.load('D:/model_outputs/proxy_anchor/training/note_families_front/babbling-tree-333/true_validation/pretty good/note_families_front_embeddings.npy')
    labels = np.load('D:/model_outputs/proxy_anchor/training/note_families_front/babbling-tree-333/true_validation/pretty good/note_families_front_circ_labels.npy', allow_pickle=True)
    tsne = TSNE(n_components=3, verbose=0, perplexity=30)
    embeddings = l2_norm(data)

    z = tsne.fit_transform(embeddings)
    array = np.array([True if 'G50' in lab or idx in [911, 912, 913] else False for idx, lab in enumerate(labels)])

    data = np.hstack([z[array, :], labels.reshape(-1, 1)[array, :]])
    return data, np.hstack([z[~array, :], labels.reshape(-1, 1)[~array, :]])


if __name__ == '__main__':
    dataPath = ''
    #data = np.load(dataPath)
    data, data_other = createRandomClusterData()

    # Data is rows of [x,y,z,labels] (x,y,z - floats) (labels - floats / ints)
    mapDataAsPointCloud(data, data_other)
