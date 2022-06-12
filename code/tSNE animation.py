import pickle
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import mplcursors

import numpy as np
from utils import parse_arguments, get_front_back_seal


def parse(name):
    pack = name.split("_")[1]
    note = name.split("note_")[-1][0::]

    if sys.platform != 'linux':
        if pack == 'G':
            return f'D:/raw_data/genuines/Pack_100_4/{note}/{note}_RGB_0.bmp'
        if pack == 'Gsmall':
            return f'D:/raw_data/1604_data/1604_notes/Pack_{pack}/{note}/{note}_RGB_0.bmp'
        return f'D:/raw_data/1604_data/1604_notes/Pack_{pack}/{note}/{note}_RGB_Front.bmp'
    else:
        if pack == 'G':
            return f'/mnt/ssd1/Genesys_2_Capture/genuine/100_4/{note}/{note}_RGB_0.bmp'
        if pack == 'Gsmall':
            return f'/mnt/ssd1/Genesys_2_Capture/counterfeit/Pack_{pack}/{note}/{note}_RGB_0.bmp'
        return f'/mnt/ssd1/Genesys_2_Capture/counterfeit/Pack_{pack}/{note}/{note}_RGB_Front.bmp'


def parse_cars(param):
    return '../data/cars196/car_ims/' + param


def parse_cub(param):
    search = '_'.join([i for i in param.split('_') if not any([True if j.isdigit() else False for j in split(i)])])
    for i in os.listdir('../data/CUB_200_2011/images/'):
        if search.lower() in i.lower():
            str = f'../data/CUB_200_2011/images/{i}/'
            for j in os.listdir(str):
                if j.lower().startswith(param.lower()):
                    str += j
                    return str


def split(word):
    return [char for char in word]


def parse_paper(param):
    pass


def parse_note_styles(param):
    book = param.split('book_')[1][0:1]
    note = param.split('note_')[-1][0::]
    return f'D:/raw_data/rupert_book/Book {book}/{note}/{note}_RGB_0.bmp'


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    # Calculate, based on the axis extent, a reasonable distance
    # from the actual point in which the click has to occur (in this case 5%)
    ax = plt.gca()
    dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    #root =
    global x, y
    # Check for every point if the click was close enough:
    bench = 9999
    for i in range(len(x)):
        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
            dist = abs(np.sqrt((x[i] - ix) ** 2 + (y[i] - iy) ** 2))
            if dist < bench:
                bench = dist
                i_close = i

    print(f'Clicked on {im_paths[i_close]}')
    if ds.startswith('note_families_'):
        name = parse(im_paths[i_close])

        if ds == 'note_families_seal':
            note_image, back_note_image, seal, df = get_front_back_seal(name, maskrcnn, True,
                                                                        ds == 'note_families_seal')
            img = cv2.rotate(seal, cv2.ROTATE_180)

        if ds == 'note_families_back':
            img = cv2.imread(name.replace('Front', 'Back'))
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if ds == 'note_families_front':
            img = cv2.imread(name)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'pack_0' in im_paths[i_close]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        fig = plt.figure()
        plt.title(im_paths[i_close])
        fig1 = fig.add_subplot(1, 1, 1)
        fig1.imshow(img)
        plt.show(block=False)

    if ds.startswith('paper'):
        name = parse(im_paths[i_close])
        note_image, back_note_image, seal, df = get_front_back_seal(name, maskrcnn, True, True)

        scaleY = note_image.shape[0] / 512
        scaleX = note_image.shape[1] / 1024

        if not df[df['className'] == 'FedSeal']['roi'].empty:
            fed_roi = df[df['className'] == 'FedSeal']['roi'].values[0]
            paper = note_image[int(round((fed_roi[2] + random.choice(np.arange(15, 25, 1))) * scaleY)):
                        int(round((fed_roi[2] + random.choice(np.arange(30, 50, 1))) * scaleY)),
                        int(round((fed_roi[3] - random.choice(np.arange(18, 30, 1))) * scaleX)):
                        int(round((fed_roi[3] - random.choice(np.arange(0, 10, 1))) * scaleX))]
            paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
            fig = plt.figure()
            plt.title(im_paths[i_close])
            fig1 = fig.add_subplot(1, 1, 1)
            fig1.imshow(paper)
            plt.show(block=False)

    if ds.startswith('cars'):
        name = parse_cars(im_paths[i_close])
        img = cv2.imread(name)
        fig = plt.figure()
        fig1 = fig.add_subplot(1, 1, 1)
        plt.title(im_paths[i_close])
        fig1.imshow(img)
        plt.show(block=False)

    if ds.startswith('cub'):
        name = parse_cub(im_paths[i_close])
        img = cv2.imread(name)
        fig = plt.figure()
        fig1 = fig.add_subplot(1, 1, 1)
        plt.title(im_paths[i_close])
        fig1.imshow(img)
        plt.show(block=False)

    if ds.startswith('note_styles'):
        name = parse_note_styles(im_paths[i_close])

        img = cv2.imread(name)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = plt.figure()
        plt.title(im_paths[i_close])
        fig1 = fig.add_subplot(1, 1, 1)
        fig1.imshow(img)
        plt.show(block=False)

def main():
    root_dir = 'D:/model_outputs/proxy_anchor/training/'
    global ds, x, y, im_paths
    ds = args.dataset

    default_models = {'cars': 'glowing-sun-314',
                      'cub': 'pleasant-donkey-171',
                      'note_families_front': 'ancient-brook-119',
                      'note_families_back': 'gentle-river-20',
                      'note_families_seal': 'laced-totem-16',
                      'note_styles' : 'fresh-pond-138',
                      'paper': 'fanciful-bee-10'}

    default_generator = {'cars': 'validation',
                         'cub': 'validation',
                         'note_families_front': 'test',
                         'note_families_back': 'test',
                         'note_families_seal': 'test',
                         'note_styles': 'test',
                         'paper': 'validation'}

    if args.model_name is None:
        model_name = default_models[args.dataset]
    else:
        model_name = args.model_name

    if args.gen is None:
        generator = default_generator[args.dataset]
    else:
        generator = args.gen

    tSNE_plots = []
    for (root, dirs, files) in os.walk(root_dir):
        for file in files:
            if (generator in root) and (ds in root) and (model_name in root) and ('truth_fine_tSNE.pkl' in file):
                x = os.path.join(root,  file)
                if x.split('\\')[-3].isdigit() or generator == 'true_validation':
                    tSNE_plots.append(x)

    if len(tSNE_plots) > 1:
        tSNE_plots = sorted(tSNE_plots, key=lambda x: int(x.split('\\')[-3])) # Sort by epoch
    plt.figure()
    plt.close()

    for idx, x in enumerate(tSNE_plots):
        fig = pickle.load(open(x, 'rb'))
        plt.title(f"{model_name} {ds} - {generator}, epoch {int(idx*5)}/{int(len(tSNE_plots)*5 - 5)}")
        mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(
            sel.artist.annots[sel.target.index]))
        # mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(
        #   sel.artist.im_paths[sel.target.index]))
        import matplotlib
        aaa = fig.axes[0].get_children()
        for obj in aaa:
            if isinstance(obj, matplotlib.collections.PathCollection):
                im_paths = obj.im_paths
                x, y = zip(*obj.get_offsets())
                break
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.rcParams['keymap.quit'].append(' ')
        plt.show(block=True)


if __name__ == '__main__':
    args = parse_arguments()
    if args.dataset == 'paper' or args.dataset == 'note_families_seal':
        from maskrcnn import MaskRCNN
        maskrcnn = MaskRCNN()
    main()