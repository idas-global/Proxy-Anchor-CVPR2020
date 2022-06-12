import json

import cv2
import matplotlib.pyplot as plt

import os
import numpy
from tqdm import tqdm

from augment_paper import get_notes_per_family, get_valid_notes, create_dirs
import sys
import json
from utils import get_front_back_seal


def main():
    flipped_dict = {}
    notes_per_family = get_notes_per_family(location_1604_notes, location_genuine_notes)
    k = 0
    for circ_key, notes_frame in tqdm(notes_per_family, desc='Unique Family'):
        pnt_key = notes_frame["parent note"].values[0]

        if pnt_key == 'NO DATA':
            pnt_key = circ_key
            if pnt_key == 'NO DATA':
                continue

        if pnt_key != 'NOLABEL':
            continue

        valid_notes = get_valid_notes(location_genuine_notes, location_1604_notes, notes_frame, specs_wanted,
                                      sides_wanted)
        for iter, (side, spec, pack, note_num, note_dir) in tqdm(enumerate(valid_notes),
                                                                 desc=f'{len(valid_notes)} Originals'):
            k += 1
            if k > 3:
                break
            note_image, back_note_image, seal, df = get_front_back_seal(note_dir, None, False, False)
            note_image = cv2.resize(note_image, (200, 200))
            plt.imshow(note_image)
            plt.rcParams['keymap.quit'].append('q')
            plt.rcParams['keymap.quit'].append('p')
            plt.show(block=True)

            ans = 'gibberish'
            while ans != 'q' and ans != 'p':
                ans = input('Is the Note Flipped')
                if ans == 'q':
                    flipped_dict[note_dir] = False
                if ans == 'p':
                    flipped_dict[note_dir] = True


    with open('./flipped_dict.json', 'w+') as handle:
        json.dump(flipped_dict, handle)
    return flipped_dict


if __name__ == '__main__':
    if sys.platform == 'linux':
        location_1604_notes = '/mnt/ssd1/Genesys_2_Capture/counterfeit/'
        location_genuine_notes = '/mnt/ssd1/Genesys_2_Capture/genuine/100_4/'
        aug_location_1604_fronts = '/mnt/ssd1/Genesys_2_Capture/1604_fronts_augmented/'
        aug_location_1604_backs = '/mnt/ssd1/Genesys_2_Capture/1604_backs_augmented/'
        aug_location_1604_seals = '/mnt/ssd1/Genesys_2_Capture/1604_seals_augmented/'
        aug_location_1604_paper = '/mnt/ssd1/paper_samples/'
    else:
        location_1604_notes = 'D:/raw_data/1604_data/1604_notes/'
        location_genuine_notes = 'D:/raw_data/genuines/Pack_100_4/'
        aug_location_1604_fronts = 'D:/raw_data/1604_data/1604_fronts_augmented/'
        aug_location_1604_backs = 'D:/raw_data/1604_data/1604_backs_augmented/'
        aug_location_1604_seals = 'D:/raw_data/1604_data/1604_seals_augmented/'
        aug_location_1604_paper = 'D:/raw_data/1604_data/1604_paper_augmented/'

    sides_wanted = ['Front']  # (0 / 1)
    specs_wanted = ['RGB']
    main()