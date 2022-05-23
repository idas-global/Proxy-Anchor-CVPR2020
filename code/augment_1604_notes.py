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
    global_csv = form_1604_frame()

    genuine_frame = form_genuine_frame()

    global_csv = pd.concat((global_csv, genuine_frame))

    empty_aug_dir()

    notes_per_family = global_csv.groupby(['circular 1'])
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]

        if pnt_key == 'NO DATA': pnt_key = circ_key

        dest = get_filepath(aug_location_1604_notes, f'{pnt_key}_{circ_key}')
        valid_notes = []

        for idx, note in notes_frame.iterrows():
            if isinstance(note, pd.Series): # Consistent Datatype
                note = pd.DataFrame(note).T

            missing_per_frame = 0

            for side in sides_wanted:
                for spec in specs_wanted:
                    pack = note['pack'].values[0]
                    note_num = str(note['pack position'].values[0])
                    root_loc = f'{location_1604_notes}Pack_{pack}/'
                    if pack == 'G':
                        root_loc = location_genuine_notes

                    if not os.path.exists(f'{root_loc}{note_num}/{note_num}_{spec}_{side}.bmp'):
                        print(f'{pack} {note_num} missing')
                        missing_per_frame += 1
                        continue
                    valid_notes.append((side, spec, pack, note_num,
                                        f'{location_1604_notes}Pack_{pack}/{note_num}/{note_num}_{spec}_{side}.bmp'))

        if len(valid_notes) > 0:
            iters = aug_fac - len(valid_notes)
            extra_notes_per_note = iters/len(valid_notes)

            for iter, (side, spec, pack, note_num, note_dir) in tqdm(enumerate(valid_notes), desc=f'{len(valid_notes)} Originals'):
                os.makedirs(dest, exist_ok=True)

                root_loc = f'{location_1604_notes}Pack_{pack}/'
                if pack == 'G':
                    root_loc = location_genuine_notes

                note_object = ImageBMP(f'{root_loc}{note_num}/{note_num}_{spec}_{side}.bmp',
                                       straighten=True, rotation=180)
                note_image = note_object.array

                if extra_notes_per_note >= 1:
                    iters = int(np.ceil(extra_notes_per_note))
                else:
                    iters = 1
                    if np.isclose(iter * extra_notes_per_note, 1):
                        iters = 2

                aug_obj = augment()
                for aug_num in range(iters):

                    aug_key = note_num + '0000' + str(aug_num)
                    aug_image = aug_obj(image=note_image)['image']
                    # plt.imshow(aug_image)
                    # plt.show()
                    aug_image = cv2.resize(aug_image, (int(aug_image.shape[1] / 10), int(aug_image.shape[0] / 10)))
                    cv2.imwrite(dest + f'/{aug_key}_{spec}_{side}.bmp', aug_image)


def form_genuine_frame():
    genuine_frame = []
    for i in [location_genuine_notes + i for i in os.listdir(location_genuine_notes)
              if os.path.isdir(location_genuine_notes + i)]:
        genuine_frame.append({
            'pack position'       : int(os.path.split(i)[-1]),
            'serial number'       : 'PLACEHOLDER',
            'date of activity'    : pd.NaT,
            'zip code bank'       : 'PLACEHOLDER',
            'zip code original'   : 'PLACEHOLDER',
            'bank name'           : 'PLACEHOLDER',
            'bank routing number' : np.nan,
            'circular 1'          : 'GENUINE',
            'parent note'         : 'GENUINE',
            'originallat'         : np.nan,
            'originallng'         : np.nan,
            'banklat'             : np.nan,
            'banklng'             : np.nan,
        })
    genuine_frame = pd.DataFrame(genuine_frame)
    genuine_frame['pack'] = 'G'
    return genuine_frame


def form_1604_frame():
    list_of_csvs = []
    for pack_1604 in [location_1604_notes + "1604 Data/" + i for i in os.listdir(location_1604_notes + r'1604 Data/') if
                      i.startswith('PACK_')]:
        pack = pd.read_csv(pack_1604)
        pack.columns = [i.lower() if i != 'Pack Positon' else 'pack position' for i in pack.columns]
        pack['pack'] = pack_1604.split('_')[-1].split('.')[0]
        list_of_csvs.append(pack)
    global_csv = pd.concat(list_of_csvs, axis=0).reset_index(drop=True)
    del global_csv['circular 2']
    global_csv['circular 1'] = [circ.replace(' ', '').replace('C', '').replace('PN', '').replace('-', '')
                                if not isinstance(circ, np.float) else 'NO DATA' for circ in global_csv['circular 1']]
    global_csv['parent note'] = [circ.replace(' ', '').replace('C', '').replace('PN', '').replace('-', '')
                                 if not isinstance(circ, np.float) else 'NO DATA' for circ in global_csv['parent note']]
    return global_csv


def get_filepath(location_1604_notes, circ_key):
    return location_1604_notes + str(circ_key)

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
    if os.path.exists(aug_location_1604_notes):
        shutil.rmtree(aug_location_1604_notes)
    os.makedirs(aug_location_1604_notes)


def get_valid_dirs():
    return [note for note in os.listdir(location_1604_notes) if note.isdigit() and os.path.isdir(location_1604_notes + note)]


if __name__ == '__main__':
    location_1604_notes = 'D:/1604_notes/'
    location_genuine_notes = 'D:/genuines/Pack_100_4/'
    aug_location_1604_notes = 'D:/1604_notes_augmented/'

    sides_wanted = ['Front'] # (0 / 1)
    specs_wanted = ['RGB']
    aug_fac = 5
    # TODO make it work for non rgb/nir
    main()
