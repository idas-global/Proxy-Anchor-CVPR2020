import argparse, os
import urllib

import utils, losses
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from train import configure_parser, create_generators, create_save_dir, create_and_compile_model
from generator import NoteStyles, Cars

import tensorflow_addons as tfa
import tensorflow_hub as hub

def main():
    args = configure_parser()

    os.chdir('../data/')
    data_root = os.getcwd()
    # Dataset Loader and Sampler

    seed = np.random.choice(range(144444))
    train_gen, val_gen, test_gen = create_generators(args, seed)

    save_path = create_save_dir(args)
    model_dir = save_path + './untrained_model.h5'

    model, criterion = create_and_compile_model(train_gen, args)

    model.load_weights(checkpoint_path)

