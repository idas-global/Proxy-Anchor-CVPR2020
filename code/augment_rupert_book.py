import os
import shutil
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as aug
import pandas as pd
from tqdm import tqdm

from noteclasses import ImageBMP


def main():
    rupert_data = pd.read_csv(rupert_location + 'rupert_pack_order.csv')
    rupert_data['series'] = rupert_data['series'].map(lambda x : x.lower().replace('series', '').strip())
    rupert_data.index = rupert_data.index.astype(str)

    rupert_notes = get_valid_dirs()
    assert len(rupert_notes) == len(rupert_data)

    #empty_aug_dir()

    notes_per_denom_and_series = rupert_data.groupby(['denom', 'series'])
    for key, notes_frame in tqdm(notes_per_denom_and_series, desc='Unique Series and Denom Pairs'):
        denom_key, series_key = key
        if denom_key != 100:
            continue

        dest = get_filepath(aug_rupert_location, denom_key, series_key)
        os.makedirs(dest, exist_ok=True)

        for note, frames in notes_frame.iterrows():
            if isinstance(frames, pd.Series): # Consistent Datatype
                frames = pd.DataFrame(frames).T

            for side in sides_wanted:
                for spec in specs_wanted:
                    note_object = ImageBMP(rupert_location + note + f'/{note}_{spec}_{side}.bmp', straighten=True,
                                           rotation=180)
                    note_image = note_object.array

                    if len(frames) < aug_fac:
                        iters = aug_fac - len(frames)
                        if iters <= 0:
                            os.makedirs(dest, exist_ok=True)
                            cv2.imwrite(dest + f'/{note}_{spec}_{side}.bmp', note_image)
                        else:
                            aug_obj = augment()
                            for aug_num in range(iters):
                                aug_key = note + '0000' + str(aug_num)
                                aug_image = aug_obj(image=note_image)['image']
                                # plt.imshow(aug_image)
                                # plt.show()
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


def empty_aug_dir():
    if os.path.exists(aug_rupert_location):
        shutil.rmtree(aug_rupert_location)
    os.makedirs(aug_rupert_location)


def get_valid_dirs():
    return [note for note in os.listdir(rupert_location) if note.isdigit() and os.path.isdir(rupert_location + note)]


if __name__ == '__main__':
    rupert_location = 'D:/Rupert_Book_Captures/'
    aug_rupert_location = 'D:/Rupert_Book_Augmented/'
    sides_wanted = [0] # (0 / 1)
    specs_wanted = ['RGB']
    aug_fac = 30
    # TODO make it work for non rgb/nir
    main()
