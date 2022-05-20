import os
import shutil
import uuid
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as aug
import pandas as pd
from tqdm import tqdm

from noteclasses import ImageBMP


def main():
    rupert_notes = get_valid_dirs()
    rupert_notes = sorted(rupert_notes, key=lambda x: int(x))
    assert len(rupert_notes) == len(rupert_data)

    #empty_aug_dir()

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
                    note_object = ImageBMP(rupert_location + rupert_notes[int(note)] + f'/{rupert_notes[int(note)]}_{spec}_{side}.bmp',
                                           straighten=True,
                                           rotation=180)
                    note_image = note_object.array

                    if len(frames) < aug_fac:
                        iters = aug_fac - len(frames)
                        if iters <= 0:
                            os.makedirs(dest, exist_ok=True)
                            note_image = cv2.resize(note_image, (int(note_image.shape[1]/10), int(note_image.shape[0]/10)))
                            cv2.imwrite(dest + f'/{note}_{spec}_{side}.bmp', note_image)
                        else:
                            aug_obj = augment()
                            for aug_num in range(iters):
                                import uuid
                                aug_key = note + '0000' + str(aug_num) + '_' + str(uuid.uuid4())[0:5]
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


def empty_aug_dir():
    if os.path.exists(aug_rupert_location):
        shutil.rmtree(aug_rupert_location)
    os.makedirs(aug_rupert_location)


def get_valid_dirs():
    return [note for note in os.listdir(rupert_location) if note.isdigit() and os.path.isdir(rupert_location + note)]


if __name__ == '__main__':
    # aug_rupert_location = '/mnt/ssd1/Rupert_Book_Augmented/'
    # empty_aug_dir()
    #
    # rupert_locations = ['/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders' + f'/Book {i}/'
    #                     for i in [0, 1, 2, 4, 5, 6]]

    rupert_data = pd.read_csv('/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/'
                              + 'rupert_pack_order.csv', header=None)
    rupert_data.columns = ['series', 'denom', 'serial']
    rupert_data['series'] = rupert_data['series'].map(lambda x : x.lower().replace('series', '').strip())
    rupert_data.index = rupert_data.index.astype(str)

    # for rupert_location in rupert_locations:
    #     sides_wanted = [0] # (0 / 1)
    #     specs_wanted = ['RGB']
    #     aug_fac = 5
    #     # TODO make it work for non rgb/nir
    #     main()

    aug_rupert_location = '/mnt/ssd1/Rupert_Book_Augmented_Test/'
    empty_aug_dir()

    rupert_location = '/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders' + f'/Book 7/'

    sides_wanted = [0]  # (0 / 1)
    specs_wanted = ['RGB']
    aug_fac = 5
    # TODO make it work for non rgb/nir
    main()