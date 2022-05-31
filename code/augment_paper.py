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

from maskrcnn import MaskRCNN
from noteclasses import ImageBMP
import random


def get_notes_per_family(notes_loc, genuine_notes_loc):
    global_csv = form_1604_frame(notes_loc)
    genuine_frame = form_genuine_frame(genuine_notes_loc)
    global_csv = pd.concat((global_csv, genuine_frame))
    notes_per_family = global_csv.groupby(['circular 1'])
    return notes_per_family


def main():
    notes_per_family = get_notes_per_family(location_1604_notes, location_genuine_notes)

    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]

        if pnt_key == 'NO DATA':
            pnt_key = circ_key
            if pnt_key == 'NO DATA':
                continue

        dest_back, dest_front, dest_paper, dest_seal = create_dirs(circ_key, pnt_key)

        valid_notes = get_valid_notes(location_genuine_notes, location_1604_notes, notes_frame, specs_wanted, sides_wanted)

        if len(valid_notes) > 0:
            iters = aug_fac - len(valid_notes)
            extra_notes_per_note = iters/len(valid_notes)

            for iter, (side, spec, pack, note_num, note_dir) in tqdm(enumerate(valid_notes), desc=f'{len(valid_notes)} Originals'):
                note_image, back_note_image, seal, df = get_front_back_seal(note_dir, maskrcnn, DO_PAPER, DO_SEAL)

                if extra_notes_per_note < 0:
                    iters = 1
                else:
                    frac, iters = math.modf(extra_notes_per_note)
                    iters += 1 + random.choices(range(2), weights=[1 - frac, frac])[0]

                iters = int(iters)

                for aug_num in range(iters):
                    aug_obj = augment()
                    aug_key = note_num + '_' + str(uuid.uuid4())[0:3] + '_' + str(aug_num)

                    if DO_BACK:
                        back_aug_image = aug_obj(image=back_note_image)['image']
                        back_aug_image = cv2.resize(back_aug_image, (int(back_aug_image.shape[1] / 10),
                                                                     int(back_aug_image.shape[0] / 10)))
                        cv2.imwrite(dest_back  + f'/{aug_key}_{spec}_{side}.bmp', back_aug_image)

                    if DO_FRONT:
                        aug_image = aug_obj(image=note_image)['image']
                        aug_image_rz = cv2.resize(aug_image, (int(aug_image.shape[1] / 10), int(aug_image.shape[0] / 10)))
                        cv2.imwrite(dest_front + f'/{aug_key}_{spec}_{side}.bmp', aug_image_rz)

                    if DO_SEAL and not df[df['className'] == 'TrsSeal']['roi'].empty:
                        aug_seal = aug_obj(image=seal)['image']
                        aug_seal = cv2.resize(aug_seal, (int(aug_seal.shape[1] / 2), int(aug_seal.shape[0] / 2)))
                        cv2.imwrite(dest_seal + f'/{aug_key}_{spec}_{side}.bmp', aug_seal)

                    if DO_PAPER and not df[df['className'] == 'FedSeal']['roi'].empty:
                        scaleY = note_image.shape[0] / 512
                        scaleX = note_image.shape[1] / 1024

                        paper = get_paper_sample(df, aug_image, scaleX, scaleY)

                        if paper is not None:
                            cv2.imwrite(dest_paper + f'/{aug_key}_{spec}_{side}.bmp', paper)


def create_dirs(circ_key, pnt_key):
    dest_front = get_filepath(aug_location_1604_fronts, f'{pnt_key}_{circ_key}')
    dest_back = get_filepath(aug_location_1604_backs, f'{pnt_key}_{circ_key}')
    dest_seal = get_filepath(aug_location_1604_seals, f'{pnt_key}_{circ_key}')
    dest_paper = get_filepath(aug_location_1604_paper, f'{pnt_key}_{circ_key}')
    os.makedirs(dest_front, exist_ok=True)
    os.makedirs(dest_back, exist_ok=True)
    os.makedirs(dest_seal, exist_ok=True)
    os.makedirs(dest_paper, exist_ok=True)
    return dest_back, dest_front, dest_paper, dest_seal


def get_front_back_seal(note_dir, maskrcnn, DO_PAPER=True, DO_SEAL=True):
    note_object = ImageBMP(note_dir,
                           straighten=True, rotation=None)
    note_image = note_object.array

    if 'Front' in note_dir:
        note_object = ImageBMP(note_dir.replace('Front', 'Back'),
                               straighten=True, rotation=None)
    else:
        note_object = ImageBMP(note_dir.replace('_0.bmp', '_1.bmp'),
                               straighten=True, rotation=None)
    back_note_image = note_object.array

    df = pd.DataFrame()
    seal = None
    if DO_PAPER or DO_SEAL:
        df = maskrcnn.detect(note_image, determineOrientation=False)

        scaleY = note_image.shape[0] / 512
        scaleX = note_image.shape[1] / 1024

        df = df[~df['classID'].duplicated(keep='first')]

        if not df[df['className'] == 'TrsSeal']['roi'].empty:
            seal_roi = df[df['className'] == 'TrsSeal']['roi'].values[0]

            seal = note_image[int(round(seal_roi[0] * scaleY)):int(round(seal_roi[2] * scaleY)),
                              int(round(seal_roi[1] * scaleX)): int(round(seal_roi[3] * scaleX))]
    return note_image, back_note_image, seal, df


def get_paper_sample(df, note_image, scaleX, scaleY):
    fed_roi = df[df['className'] == 'FedSeal']['roi'].values[0]
    paper = None
    thresh_before = None
    prc_before = 0
    for filter_size in [120, 110, 100, 80, 60, 50]:
        note_image_blurred = cv2.bilateralFilter(note_image, 30, filter_size, filter_size)
        hsv_note = cv2.cvtColor(note_image_blurred, cv2.COLOR_BGR2HSV_FULL)

        paper_sample = hsv_note[int(round((fed_roi[2] + 20) * scaleY)):int(round((fed_roi[2] + 40) * scaleY)),
                       int(round((fed_roi[3] - 25) * scaleX)): int(round((fed_roi[3] - 5) * scaleX))]

        thresh = cv2.inRange(hsv_note,
                             np.array([np.min(paper_sample[:, :, i]) for i in range(paper_sample.shape[-1])]),
                             np.array([np.max(paper_sample[:, :, i]) for i in range(paper_sample.shape[-1])]))
        prc = np.sum(thresh) / (255 * thresh.shape[0] * thresh.shape[1])
        if prc > 0.25:
            a = abs(prc_before - 0.25)
            b = abs(prc - 0.25)
            if a < b and thresh_before is not None:
                paper = note_image.reshape(-1, 3)[thresh_before.ravel() == 255, :]
            else:
                paper = note_image.reshape(-1, 3)[thresh.ravel() == 255, :]
            prevN = math.floor(math.sqrt(len(paper)))
            paper = np.reshape(paper[0:prevN ** 2, :], (prevN, prevN, 3))
            paper = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV_FULL)
            paper_flat = paper.reshape(-1, 3)
            aaa = np.lexsort((paper_flat[:, 0], paper_flat[:, 1], paper_flat[:, 2]))
            sample = paper_flat[aaa].reshape(paper.shape)
            sample = cv2.cvtColor(sample, cv2.COLOR_HSV2BGR_FULL)
            paper = sample
            break

        thresh_before = thresh.copy()
        prc_before = prc
    return paper


def get_valid_notes(location_genuine_notes, location_1604_notes, notes_frame, specs_wanted, sides_wanted):
    valid_notes = []
    for idx, note in notes_frame.iterrows():
        if isinstance(note, pd.Series):  # Consistent Datatype
            note = pd.DataFrame(note).T

        missing_per_frame = 0

        for side in sides_wanted:
            for spec in specs_wanted:
                pack = note['pack'].values[0]
                note_num = str(note['pack position'].values[0])
                root_loc = f'{location_1604_notes}Pack_{pack}/'

                if pack == 'G':
                    root_loc = location_genuine_notes

                note_dir = f'{root_loc}{note_num}/{note_num}_{spec}_{side}.bmp'

                if not os.path.exists(note_dir):
                    side_2 = '1'
                    if side == 'Front':
                        side_2 = '0'

                    note_dir = f'{root_loc}{note_num}/{note_num}_{spec}_{side_2}.bmp'
                    if not os.path.exists(note_dir):
                        print('### Missing ###')
                        print(f'{root_loc}{note_num}/{note_num}_{spec}_{side_2}.bmp')
                        print(f'{root_loc}{note_num}/{note_num}_{spec}_{side}.bmp')
                        missing_per_frame += 1
                        continue

                valid_notes.append((side, spec, pack, note_num, note_dir))
    return valid_notes


def form_genuine_frame(location_genuine_notes):
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


def form_1604_frame(location_1604_notes):
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


def empty_aug_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_valid_dirs():
    return [note for note in os.listdir(location_1604_notes) if note.isdigit() and os.path.isdir(location_1604_notes + note)]


if __name__ == '__main__':
    DO_PAPER = True
    DO_SEAL = True
    DO_FRONT = True
    DO_BACK = True
    DELETE_DATA = True

    if sys.platform == 'linux':
        location_1604_notes = '/mnt/ssd1/Genesys_2_Capture/counterfeit/'
        location_genuine_notes = '/mnt/ssd1/Genesys_2_Capture/genuine/100_4/'
        aug_location_1604_fronts = '/mnt/ssd1/Genesys_2_Capture/1604_fronts_augmented/'
        aug_location_1604_backs = '/mnt/ssd1/Genesys_2_Capture/1604_backs_augmented/'
        aug_location_1604_seals = '/mnt/ssd1/Genesys_2_Capture/1604_seals_augmented/'
        aug_location_1604_paper = '/mnt/ssd1/paper_samples/'
    else:
        location_1604_notes = 'D:/1604_notes/'
        location_genuine_notes = 'D:/genuines/Pack_100_4/'
        aug_location_1604_fronts = 'D:/1604_fronts_augmented/'
        aug_location_1604_backs = 'D:/1604_backs_augmented/'
        aug_location_1604_seals = 'D:/1604_seals_augmented/'
        aug_location_1604_paper = 'D:/1604_paper_augmented/'

    if DELETE_DATA:
        time.sleep(5)
        print('SLEEPING FOR 5 SECONDS BECAUSE THIS DELETES DATASETS')
        for i in np.arange(5, 0):
            print(i)
            time.sleep(i)

        if DO_FRONT:
            empty_aug_dir(aug_location_1604_fronts)
        if DO_BACK:
            empty_aug_dir(aug_location_1604_backs)
        if DO_SEAL:
            empty_aug_dir(aug_location_1604_seals)
        if DO_PAPER:
            empty_aug_dir(aug_location_1604_paper)

    sides_wanted = ['Front'] # (0 / 1)
    specs_wanted = ['RGB']
    aug_fac = 4
    # TODO make it work for non rgb/nir
    maskrcnn = MaskRCNN()
    main()
