from tensorflow import keras
from keras.utils.np_utils import to_categorical

import numpy as np
from tqdm import tqdm


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dl_tr, shuffle=True):
        'Initialization'
        self.batch_size = dl_tr.batch_size
        self.shuffle = shuffle
        self.dataset_size = len(dl_tr.dataset.im_paths)
        self.data = dl_tr

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        x, y = next(iter(self.data))
        x = np.moveaxis((np.array(x.squeeze())), 1, -1).astype(np.float32)
        y = to_categorical(np.array(y).astype(np.float32).reshape(-1, 1))
        return x, y
