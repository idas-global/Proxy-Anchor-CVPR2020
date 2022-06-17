import numpy as np
import os
from utils import parse_arguments


def main(dataset):

    tSNE_plots = []
    for (root, dirs, files) in os.walk(f'../logs/{dataset}/'):
        for file in files:
            if (model_name in root) and ('val_X.npy' in file if generator == 'validation' else 'eval_X.npy' in file):
                x = os.path.join(root, file)
                if all(i.isdigit() for i in root.split('\\')[-2].split('-')[-1]):
                    tSNE_plots.append(x)

    if len(tSNE_plots) > 1:
        tSNE_plots = sorted(tSNE_plots, key=lambda x: int(x.split('\\')[-3]))  # Sort by epoch

if __name__ == '__main__':
    args = parse_arguments()

    default_models = {'cars': 'glowing-sun-314',
                      'cub': 'pleasant-donkey-171',
                      'note_families_front': 'babbling-tree-333',
                      'note_families_back': 'gentle-river-20',
                      'note_families_seal': 'laced-totem-16',
                      'note_styles' : 'fresh-pond-138',
                      'paper': 'fanciful-bee-10'}

    default_generator = {'cars': 'validation',
                         'cub': 'validation',
                         'note_families_front': 'test',
                         'note_families_back': 'test',
                         'note_families_seal': 'test',
                         'note_styles': 'validation',
                         'paper': 'validation'}

    if args.model_name is None:
        model_name = default_models[args.dataset]
    else:
        model_name = args.model_name

    if args.gen is None:
        generator = default_generator[args.dataset]
    else:
        generator = args.gen

    main(args.dataset)


