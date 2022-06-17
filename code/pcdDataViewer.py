import cv2
import cmapy
import random
import numpy as np
import open3d as o3d
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from utils import l2_norm


def to_uint8(arr):
    return (255*(arr - arr.min()) / arr.max()).astype(np.uint8)


def mapLabelsToColours(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return cv2.applyColorMap(to_uint8(labels), cmapy.cmap('jet')) / 255.0


def mapDataAsPointCloud(data):
    pts    = data[:, :3]
    labels = data[:, 3]
    colors = mapLabelsToColours(labels)

    pcd = o3d.geometry.PointCloud()

    pcd.colors = o3d.utility.Vector3dVector(colors[:,0,:])
    pcd.points = o3d.utility.Vector3dVector(pts)

    o3d.visualization.draw([pcd], show_ui=True)


def createRandomClusterData(n=1200):
    data = np.load('D:/model_outputs/proxy_anchor/training/note_families_front/babbling-tree-333/true_validation/pretty good/note_families_front_embeddings.npy')
    labels = np.load('D:/model_outputs/proxy_anchor/training/note_families_front/babbling-tree-333/true_validation/pretty good/note_families_front_circ_labels.npy', allow_pickle=True)
    tsne = TSNE(n_components=3, verbose=0, perplexity=30)
    embeddings = l2_norm(data)

    z = tsne.fit_transform(embeddings)

    data = np.hstack([z, labels.reshape(-1, 1)])
    return data

if __name__ == '__main__':
    dataPath = ''
    #data = np.load(dataPath)
    data = createRandomClusterData()

    # Data is rows of [x,y,z,labels] (x,y,z - floats) (labels - floats / ints)
    mapDataAsPointCloud(data)
