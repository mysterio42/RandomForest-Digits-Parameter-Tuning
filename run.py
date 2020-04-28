import numpy as np

from utils.data import load_data
from utils.model import train_model
from utils.plot import plot_graph
import argparse

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

    model = train_model(features, labels,args)

    plot_graph(model, features, labels,'RandomForest-gs-graph')
