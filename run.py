import argparse

import numpy as np

from utils.data import load_data, to_markdown
from utils.model import train_model
from utils.plot import plot_graph, plot_pca


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()

    parser.add_argument('--gs', type=str2bool, default=False,
                        help='Find optimal parameters with 10-Fold GridSearchCV')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1)

    args = parse_args()

    features, labels = load_data()

    to_markdown(features, labels)

    plot_pca(features.to_numpy(), labels.to_numpy())

    model = train_model(features, labels, args)

    plot_graph(model, features, labels, 'RandomForest-gs-graph')
