import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import export_graphviz

from utils.data import fet_lab_names

FIGURES_DIR = 'figures/'
GRAPH_DIR = 'graph/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_cm(cm, name):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.title(name)
    plt.savefig(FIGURES_DIR + f'Figure_{name}' + '.png')
    plt.show()


def plot_graph(model, features, labels, graph_name):
    param = '-Tsvg'
    dot_path = GRAPH_DIR + graph_name + '.dot'
    svg_path = FIGURES_DIR + graph_name + '.svg'

    feature_names, label_names = fet_lab_names(features, labels)

    export_graphviz(model, out_file=dot_path,
                    feature_names=feature_names, class_names=label_names,
                    rounded=True, filled=True,
                    precision=2, proportion=True)

    os.system(f'dot {param} {dot_path} -o {svg_path}')
