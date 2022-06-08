import pickle
import os

import cv2
import matplotlib.pyplot as plt
import mplcursors


def parse(name):
    pack = name.split("pack_")[-1][0:1]
    note = name.split("note_")[-1][0::]

    if pack == '1' and int(note) > 8:
        pass
    if pack == 'G':
        return f'E:/Genesys_2_Capture/genuine/100_4/{note}/{note}_RGB_0.bmp'
    return f'E:/Genesys_2_Capture/counterfeit/Pack_{pack}/{note}/{note}_RGB_Front.bmp'


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


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print("I clicked at x={0:5.2f}, y={1:5.2f}".format(ix,iy))

    # Calculate, based on the axis extent, a reasonable distance
    # from the actual point in which the click has to occur (in this case 5%)
    ax = plt.gca()
    dx = 5*0.010449 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 5*0.009607 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    #root =
    global x, y
    # Check for every point if the click was close enough:
    bench = 9999
    for i in range(len(x)):
        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
            dist = abs((x[i] - ix) ** 2 + (y[i] - iy) ** 2)
            if dist < bench:
                bench = dist
                i_close = i

    if ds.startswith('note_families_') or ds.startswith('paper_') :
        name = parse(im_paths[i_close])
        img = cv2.imread(name)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow(im_paths[i_close], cv2.resize(img, (1600, 800)))
    if ds.startswith('cars'):
        name = parse_cars(im_paths[i_close])
        img = cv2.imread(name)
        cv2.imshow(im_paths[i_close], cv2.resize(img, (1200, 1200)))
    if ds.startswith('cub'):
        name = parse_cub(im_paths[i_close])
        img = cv2.imread(name)
        cv2.imshow(im_paths[i_close], cv2.resize(img, (1200, 1200)))

def main():
    root_dir = 'D:/model_outputs/proxy_anchor/training/'
    global ds, x, y, im_paths

    for ds in ['cars', 'cub', 'note_families_front', 'note_families_back', 'note_families_seal', 'note_styles', 'paper']:
        for model_name in ['radiant-paper-7', 'blooming-pine-134', 'tough-yogurt-5', 'pleasant-donkey-171', 'glowing-sun-314', 'elated-pyramid-116', 'magic-wave-15', 'dutiful-snowflake-17']:
            generator = 'test'
            if ds == 'cars' or ds == 'cub':
                generator = 'validation'

            tSNE_plots = []
            for (root, dirs, files) in os.walk(root_dir):
                for file in files:
                    if (generator in root) and (ds in root) and (model_name in root) and ('truth_fine_tSNE.pkl' in file):
                        tSNE_plots.append(os.path.join(root,  file))

            tSNE_plots = sorted(tSNE_plots, key=lambda x: int(x.split('\\')[-3])) # Sort by epoch
            plt.figure()
            plt.close()

            # fig = pickle.load(open(f'D:/model_outputs/proxy_anchor/applied_models/back/tSNE.pkl', 'rb'))
            # mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(
            #     sel.artist.annots[sel.target.index]))
            # import matplotlib
            # aaa = fig.axes[0].get_children()
            # for obj in aaa:
            #     if isinstance(obj, matplotlib.collections.PathCollection):
            #         global x, y, im_paths
            #         im_paths = obj.im_paths
            #         x, y = zip(*obj.get_offsets())
            #         break
            # print(x)
            # cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # plt.show(block=True)

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
    main()