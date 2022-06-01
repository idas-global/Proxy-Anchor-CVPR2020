import math
import os
import random
import shutil
import sys
import time
import uuid
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as aug
import pandas as pd
from tqdm import tqdm

from augment_paper import empty_aug_dir
from noteclasses import ImageBMP


def main():
    rupert_data = pd.read_csv(csv_loc, header=None)
    rupert_data.columns = ['series', 'denom', 'serial']
    rupert_data['series'] = rupert_data['series'].map(lambda x : x.lower().replace('series', '').strip())
    rupert_data.index = rupert_data.index.astype(str)

    rupert_notes = get_valid_dirs()
    if len(rupert_notes) != len(rupert_data):
        print(f'WARNING: {rupert_location} only has {len(rupert_notes)}, expecting {len(rupert_data)}')
        print(f'WARNING: Assuming Consecutive folders')


    notes_per_denom_and_series = rupert_data.groupby(['denom', 'series'])
    for key, notes_frame in tqdm(notes_per_denom_and_series, desc='Unique Series and Denom Pairs'):
        denom_key, series_key = key

        dest = get_filepath(aug_rupert_location, denom_key, series_key)
        os.makedirs(dest, exist_ok=True)

        for note, frames in notes_frame.iterrows():
            if isinstance(frames, pd.Series): # Consistent Datatype
                frames = pd.DataFrame(frames).T

            for side in sides_wanted:
                for spec in specs_wanted:
                    note_dir = rupert_location + note + f'/{note}_{spec}_{side}.bmp'
                    if not os.path.exists(note_dir):
                        if side == 0:
                            note_dir = rupert_location + note + f'/{note}_{spec}_Front.bmp'
                        else:
                            note_dir = rupert_location + note + f'/{note}_{spec}_Back.bmp'
                        if not os.path.exists(note_dir):
                            print('### Missing ###')
                            print(note_dir)
                            continue

                    note_object = ImageBMP(note_dir, straighten=True,
                                           rotation=180)
                    note_image = note_object.array

                    iters = aug_fac - len(frames)
                    extra_notes_per_note = iters / len(frames)

                    if extra_notes_per_note < 0:
                        iters = 1
                    else:
                        frac, iters = math.modf(extra_notes_per_note)
                        iters += 1 + random.choices(range(2), weights=[1 - frac, frac])[0]

                    iters = int(iters)
                    book = rupert_location.split('/')[-2].split(' ')[-1]
                    for aug_num in range(iters):
                        aug_obj = augment()

                        aug_key = f'book_{book}_note_{note}_aug_{aug_num}_{str(uuid.uuid4())[0:4]}'
                        aug_image = aug_obj(image=note_image)['image']
                        # plt.imshow(aug_image)
                        # plt.show()
                        aug_image = cv2.resize(aug_image, (int(aug_image.shape[1] / 10), int(aug_image.shape[0] / 10)))
                        cv2.imwrite(dest + f'/{aug_key}_{spec}_{side}.bmp', aug_image)


def get_filepath(rupert_location, denom_key, prsd_series_key):
    return rupert_location + str(denom_key) + '_' + prsd_series_key + '/'

def augment():
    transform = aug.Compose([
        aug.HorizontalFlip(p=0.25),
        aug.VerticalFlip(p=0.25),
        aug.GaussNoise(p=0.15),
        aug.GaussianBlur(p=0.15),
        aug.RandomBrightnessContrast(p=0.2),
        aug.RandomShadow(p=0.2),
        aug.RandomRain(p=0.2)
    ], p=1)
    return transform


def get_valid_dirs():
    return [note for note in os.listdir(rupert_location) if note.isdigit() and os.path.isdir(rupert_location + note)]


if __name__ == '__main__':
    DELETE_DATA = True

    if sys.platform != 'linux':
        list_of_book_locations = ['D:/raw_data/rupert_book/Book 3/', 'D:/raw_data/rupert_book/Book 0/']
        aug_rupert_locations = ['D:/raw_data/rupert_book/rupert_book_augmented/', 'D:/raw_data/rupert_book/rupert_book_augmented_test/']
        csv_loc = 'D:/raw_data/rupert_book/rupert_pack_order.csv'
    else:
        list_of_book_locations = [f'/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/Book {i}/' for i in range(8)]
        aug_rupert_locations = ['/mnt/ssd1/Genesys_2_Capture/rupert_book_augmented/' if i < 5 else '/mnt/ssd1/Genesys_2_Capture/rupert_book_augmented_test/' for i in range(8)]
        csv_loc = '/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/rupert_pack_order.csv'

    if DELETE_DATA:
        time.sleep(5)
        print('SLEEPING FOR 5 SECONDS BECAUSE THIS DELETES DATASETS')
        for i in np.arange(5, 0, -1):
            print(i)
            time.sleep(1)
        time.sleep(5)

        for dirr in aug_rupert_locations:
            empty_aug_dir(dirr)

    sides_wanted = [0] # (0 / 1)
    specs_wanted = ['RGB']
    aug_fac = 3
    # TODO make it work for non rgb/nir

    for rupert_location, aug_rupert_location in zip(list_of_book_locations, aug_rupert_locations):
        main()
