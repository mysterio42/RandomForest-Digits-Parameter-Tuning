import glob
import os
import random
import re
import string
from operator import itemgetter

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

from utils.config import param_grid
from utils.plot import plot_cm

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model():
    """

    :param path: weight path
    :return: load model based on the path
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model, name):
    model_name = WEIGHTS_DIR + name + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(features, labels,args):
    if args.gs:

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

        gs = GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt', criterion='entropy'),
                          param_grid=param_grid, cv=10)
        gs.fit(features_train, labels_train)

        dec_trees = RandomForestClassifier(max_features='sqrt',
                                           n_estimators=gs.best_params_[
                                               'n_estimators'] if 'n_estimators' in param_grid else 100,
                                           max_depth=gs.best_params_['max_depth'] if 'max_depth' in param_grid else None,
                                           min_samples_leaf=gs.best_params_[
                                               'min_samples_leaf'] if 'min_samples_leaf' in param_grid else 1
                                           )

        cv_results = cross_validate(dec_trees, features_train, labels_train,
                                    cv=10, return_estimator=True)
        estimator_test_score = zip(list(cv_results['estimator']), cv_results['test_score'])
        model, score = max(estimator_test_score, key=itemgetter(1))

        preds = model.predict(features_test)

        cm = confusion_matrix(labels_test, preds)
        score = accuracy_score(labels_test, preds)

        best_params = re.sub("[{}' ,()]", '', str(gs.best_params_))

        plot_cm(cm, f'cm-accuracy:{score:.2f}{best_params}RandomForest-gs')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model.estimators_[0], 'RandomForest-gs-')

        return model.estimators_[0]
