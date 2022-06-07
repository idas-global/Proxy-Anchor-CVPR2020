import pickle
import os
import matplotlib.pyplot as plt
import mplcursors


def main():
    root_dir = 'D:/model_outputs/proxy_anchor/training/'
    for ds in ['cars', 'cub', 'note_families_front', 'note_families_back', 'note_families_seal', 'note_styles', 'paper']:
        for model_name in ['blooming-pine-134', 'tough-yogurt-5', 'pleasant-donkey-171', 'glowing-sun-314', 'elated-pyramid-116', 'magic-wave-15', 'dutiful-snowflake-17']:
            for generator in ['test']:
                tSNE_plots = []
                for (root, dirs, files) in os.walk(root_dir):
                    for file in files:
                        if (generator in root) and (ds in root) and (model_name in root) and ('truth_fine_tSNE.pkl' in file):
                            tSNE_plots.append(os.path.join(root,  file))

                tSNE_plots = sorted(tSNE_plots, key=lambda x: int(x.split('\\')[-3])) # Sort by epoch
                plt.figure()
                plt.close()
                for idx, x in enumerate(tSNE_plots):
                    fig = pickle.load(open(x, 'rb'))
                    plt.title(f"{model_name} {ds} - {generator}, epoch {int(idx*5)}/{int(len(tSNE_plots)*5 - 5)}")
                    mplcursors.cursor(fig, hover=True).connect("add", lambda sel: sel.annotation.set_text(sel.artist.annots[sel.target.index]))
                    mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(sel.artist.im_paths[sel.target.index]))
                    plt.rcParams['keymap.quit'].append(' ')
                    plt.show(block=True)

if __name__ == '__main__':
    main()