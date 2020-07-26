import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from tools import load_json


class Labels:
    def __init__(self, json_labels=None, group=None):
        self._log = logging.getLogger('root')
        self._log.info(f'Loading labels from {json_labels} for group: {group}')
        self._labels = load_json(json_labels)
        self._group = group
        self._group_labels = self._get_group_labels()
        self._numerical_labels, self._labels2num = self._get_numerical_labels()
        self._labels_counts = self._get_labels_counts()
        self._one_hot_labels = to_categorical(self._numerical_labels)
        self._total_count = len(self._group_labels)
        self._inverse_frequency = self._get_inverse_frequency()
        self._proportions = self._get_proportions()
        self._image2one_hot = self._get_image2one_hot()

    @property
    def image_names(self):
        return np.array(list(self._labels.keys()))

    @property
    def one_hot(self):
        return self._one_hot_labels

    @property
    def numerical(self):
        return self._numerical_labels

    @property
    def counts(self):
        return self._labels_counts

    @property
    def labels2numerical(self):
        return self._labels2num

    @property
    def categorical(self):
        return self._group_labels

    @property
    def inverse_frequency(self):
        return self._inverse_frequency

    @property
    def proportions(self):
        return self._proportions

    @property
    def image2onehot(self):
        return self._image2one_hot

    def _get_proportions(self):
        labels_num = np.sort(list(self._labels_counts.keys()))
        inverse = list()
        for num in labels_num:
            inverse.append(self._labels_counts[num] / self._total_count)
        return np.array(inverse)

    def _get_inverse_frequency(self):
        labels_num = np.sort(list(self._labels_counts.keys()))
        inverse = list()
        for num in labels_num:
            inverse.append(1 / self._labels_counts[num])
        return np.array(inverse)

    def _get_numerical_labels(self):
        label_encoder = LabelEncoder()
        numerical = label_encoder.fit_transform(self._group_labels)
        label2num = dict()
        for idx, label in enumerate(label_encoder.classes_):
            label2num[label] = idx

        return numerical, label2num

    def _get_labels_counts(self):
        unique = np.unique(self._numerical_labels)
        counts = dict()
        for unique_label in unique:
            counts[unique_label] = np.sum(self._numerical_labels == unique_label)
        return counts

    def _get_group_labels(self):
        return np.array([label[self._group] for label in self._labels.values()])

    def _get_image2one_hot(self):
        image2one_hot = dict()
        for idx, name in enumerate(self.image_names):
            image2one_hot[name] = self._one_hot_labels[idx]
        return image2one_hot


if __name__ == "__main__":
    from tools import basic_logger

    basic_logger()
    path = r'c:\DATA\SceneClassification\labels\bdd100k_labels_images_val_extracted.json'
    labels = Labels(path, 'weather')
    print(labels.image_names[:5])
    print(labels.categorical[:5])
    print(labels.counts)
    print(labels.inverse_frequency)
    print(labels.proportions)
    print(labels.labels2numerical)
    print(labels.numerical[:5])
    print(labels.one_hot[:5])
