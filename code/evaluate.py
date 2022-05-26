import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def get_overall_acc(data, coarse=False):
    key = 'fine'
    if coarse:
        key = 'coarse'
    return accuracy_score(data[f'prediction_label_{key}'], data[f'truth_label_{key}']) * 100


if __name__ == '__main__':
    training_loc = 'D:/training/golden-dawn-45/'
    ds = 'validation'

    metrics = {
        'Overall Accuracy' : get_overall_acc,
        'Overall Accuracy (Coarse)': get_overall_acc,
    }

    x = {i : [] for i in metrics.keys()}
    epochs = [int(i) for i in sorted(os.listdir(training_loc), key=lambda x: int(x))]
    for epoch in epochs:
        validation_data = pd.read_csv(training_loc + f'{epoch}/{ds}/{ds}_data.csv', index_col=0)

        key = 'fine'
        cl_report = classification_report(validation_data[f'truth_label_{key}'], validation_data[f'prediction_label_{key}'], output_dict=True)
        key = 'coarse'
        cl_report = classification_report(validation_data[f'prediction_label_{key}'], validation_data[f'truth_label_{key}'], output_dict=True)

        for key, method in metrics.items():
            x[key].append(method(validation_data, coarse='Coarse' in key))

    for key, values in x.items():
        plt.title(key)
        plt.plot(epochs, values)
