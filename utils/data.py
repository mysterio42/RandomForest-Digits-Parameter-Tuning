import pandas as pd
from sklearn import datasets

DATA_DIR = 'data/'


def load_data():
    data = datasets.load_digits()

    data_size = data.images.shape[0]

    features = data.images.reshape((data_size, -1))
    labels = data.target
    feature_size = features.shape[1]

    df = pd.DataFrame(data=features, columns=['pix' + str(el) for el in range(feature_size)])
    df['label'] = labels

    del features, labels

    return df.iloc[:, :feature_size], df.iloc[:, feature_size]


def fet_lab_names(features, labels):
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    return list(features.columns), list(map(str, list(labels.unique())))


def to_markdown(features, labels):
    features['label'] = labels
    with open(DATA_DIR + 'digits.txt', 'w') as f:
        f.write(features.head(50).to_markdown())
