import os
import shutil
import sys

import numpy as np
import pandas as pd

if sys.platform != 'linux':
    root = 'D:/raw_data/rupert_book/'
    destination = 'D:/raw_data/1604_data/1604_notes/'
    data_loc = 'D:/raw_data/1604_data/1604_notes/1604 data/'
else:
    root = '/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/'
    destination = '/mnt/ssd1/Genesys_2_Capture/counterfeit/'
    data_loc = '/mnt/ssd1/Genesys_2_Capture/counterfeit/1604 Data/'


lo_dests = [destination + i + '/' for i in ['Pack_G100small', 'Pack_G100medium', 'Pack_G100large',
                               'Pack_G50small', 'Pack_G50medium', 'Pack_G50large',
                               'Pack_G20small', 'Pack_G20medium', 'Pack_G20large',
                               'Pack_G10small', 'Pack_G10medium', 'Pack_G10large']]

book_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
book_dirs = [root + 'Book ' + str(i) + '/' if os.path.exists(root + 'Book ' + str(i) + '/') else None for i in
             book_idxs]
note_nums = {'Pack_G100small' : ['0', '56', '57', '58'],
             'Pack_G100medium' : ['59', '60', '61', '62', '63', '64', '65'],
             'Pack_G100large' : ['66', '67', '68', '69'],
             'Pack_G50small' : ['46', '47', '48', '49'],
             'Pack_G50medium' : ['50'],
             'Pack_G50large' : ['51', '52', '53', '54', '55'],
             'Pack_G20small' : ['33', '34', '35', '36', '37', '38'],
             'Pack_G20medium' : ['39', '40', '41', '42', '43', '44', '45'],
             'Pack_G20large' : [],
             'Pack_G10small' : ['21', '22', '23'],
             'Pack_G10medium' : ['24', '25', '26'],
             'Pack_G10large': ['27', '28', '29', '30', '31']
             }

for dest in lo_dests:
    note_num = note_nums[dest.split('/')[-2]]
    os.makedirs(dest, exist_ok=True)

    for idx, book in zip(book_idxs, book_dirs):
        if book is None:
            print(f'Missing {idx}')
            continue
        for note_n in note_num:
            notes = [book + i + '/' for i in os.listdir(book) if os.path.isdir(book + i)]
            notes = sorted(notes, key= lambda x: int(x.split('/')[-2]))
            path = notes[int(note_n)]
            flder = str(len(os.listdir(dest)))
            dest_path = path.replace(path.split('/')[-2], flder).replace(root, dest).replace(f"Book {idx}", "")
            os.makedirs(dest_path, exist_ok=True)

            for file in os.listdir(path):
                dest_file = file.replace(path.split('/')[-2], flder)
                print(f'{path + file}    ----> {dest_path + dest_file}')
                shutil.copy(path + file, dest_path + dest_file)
    number_of_notes = len(os.listdir('/'.join(dest_path.split('/')[0:-2])))
    dict_1604 = {
    'Pack Position' : 0,
    'Serial Number' : 'UNKNOWN',
    'Date Of Activity' : '31/03/2021',
    'ZIP Code Bank' : np.nan,
    'ZIPCode Original' : np.nan,
    'Bank Name' : np.nan,
    'Bank Routing Number': np.nan,
    'Circular 1' : dest.split('/')[-2].replace('PACK_', ''),
    'Circular 2' : np.nan,
    'Parent Note' : np.nan,
    'OriginalLat' : np.nan,
    'OriginalLng' : np.nan,
    'BankLat' : np.nan,
    'BankLng' : np.nan
     }

    list_of_dicts = []
    for i in range(number_of_notes):
        note = dict_1604.copy()
        note['Pack Position'] = i
        list_of_dicts.append(note)

    frame_1604 = pd.DataFrame(list_of_dicts)

    outpath = data_loc + dest.replace(destination, '')[0:-1] + '.csv'
    outpath = outpath.replace('Pack', 'PACK')
    print(f'Writing to {outpath}')
    frame_1604.to_csv(outpath)

