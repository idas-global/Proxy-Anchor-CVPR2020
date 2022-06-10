import os
import shutil
import sys

if sys.platform != 'linux':
    root = 'D:/raw_data/rupert_book/'
    dest = 'D:/raw_data/rupert_book/Pack_Gsmall'
else:
    root = '/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/'
    dest = '/mnt/sanshare/Datasets/notes/genesys_capture/genuine/Rupert_Binders/Pack_Gsmall/'

os.makedirs(dest, exist_ok=True)
book_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
book_dirs = [root + 'Book ' + str(i) + '/' for i in book_idxs]

for idx, book in zip(book_idxs, book_dirs):
    for note_num in ['57', '58']:
        notes = [book + i + '/' for i in os.listdir(book) if os.path.isdir(book + i)]
        notes = sorted(notes, key= lambda x: int(x.split('/')[-2]))
        path = notes[int(note_num)]
        flder = str(len(os.listdir(dest)))
        dest_path = path.replace(path.split('/')[-2], flder).replace(root, dest).replace(f"Book {idx}", "")
        os.makedirs(dest_path, exist_ok=True)

        for file in os.listdir(path):
            dest_file = file.replace(path.split('/')[-2], flder)
            print(f'{path + file}    ----> {dest_path + dest_file}')
            #shutil.copy(path + file, dest_path + dest_file)


