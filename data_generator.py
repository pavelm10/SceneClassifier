import logging
import numpy as np
import cv2
import uuid

from keras.preprocessing.image import ImageDataGenerator

import tools
from labels import Labels


class DataGenerator:

    MODES = tools.MODES
    GROUPS = tools.GROUPS

    def __init__(self, config, image_dir=None, labels_json=None, mode=None, group=None, class_counts=None):
        self.log = logging.getLogger('root')
        self.config = config
        self.image_dir = tools.str2path(image_dir)
        self.labels_json = tools.str2path(labels_json)
        self.mode = mode
        self.group = group
        self.channels = config.get('channels', 3)
        self._batch_size = config.get('flow', {}).get('batch_size', 4)
        self._validate_input()
        self.color_mode = 'rgb' if self.channels == 3 else 'grayscale'
        self.target_imsize = (self.config['target_height'], self.config['target_width'])
        self.blacklist = self.config.get('class_blacklist', [])
        self.class_counts = class_counts

        if labels_json:
            self.keras_dir = self.image_dir.parent / f'{self.image_dir.name}_{self.group}_{str(uuid.uuid1())[:8]}'
            if not self.keras_dir.exists():
                self.keras_dir.mkdir()
                self.log.info(f'Creating Keras Directory Tree - {self.mode}...')
                tools.create_keras_image_directory_tree(self.image_dir,
                                                        self.keras_dir,
                                                        self.labels_json,
                                                        self.group,
                                                        self.blacklist,
                                                        self.class_counts)
            else:
                self.log.info('Skipped Keras Directory Tree creation as it already exists')

            self.labels = Labels(labels_json, self.keras_dir, group)
        else:
            self.keras_dir = image_dir

        self._data_generator = self._init_data_gen()
        self._flow_gen = self._flow_from_directory()

    @property
    def data_generator(self):
        return self._data_generator

    @property
    def samples(self):
        return len(list(self.keras_dir.rglob("*.jpg")))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def steps_per_epoch(self):
        return np.floor(self.samples / self.batch_size)

    @property
    def flow_generator(self):
        return self._flow_gen

    @property
    def class_weights(self):
        if self.config.get('weights_type', 'frequency') == 'frequency':
            scales = self.labels.inverse_frequency
        else:
            scales = self.labels.proportions

        cls2idx = self._flow_gen.class_indices
        class_weights = dict()
        for cls, idx in cls2idx.items():
            class_weights[idx] = scales[cls]
        return class_weights

    def _validate_input(self):
        if not self.image_dir.exists():
            raise IOError(f'{self.image_dir} does not exist')

        if self.labels_json is not None and not self.labels_json.exists():
            raise IOError(f'{self.labels_json} does not exist')

        if self.group is not None and self.group not in self.GROUPS:
            raise AttributeError(f'{self.group} is unknown group, known are {self.GROUPS}')

        if self.channels != 3 and self.channels != 1:
            raise AttributeError(f'channels must be either 3 or 1')

    def _init_data_gen(self):
        cfg = self.config.get(f'{self.mode}_gen')
        if cfg:
            return ImageDataGenerator(**cfg)
        else:
            return ImageDataGenerator()

    def _flow_from_directory(self):
        cfg = self.config.get(f'flow', {})
        return self._data_generator.flow_from_directory(directory=self.keras_dir,
                                                        target_size=self.target_imsize,
                                                        color_mode=self.color_mode,
                                                        **cfg)

    def flow_from_labels(self):
        ims_in_dir = list(self.keras_dir.rglob("*.jpg"))
        ims_in_dir_names = [file.name for file in ims_in_dir]

        images = list()
        labels = list()
        image_names = list()

        while len(ims_in_dir_names):
            image_name = ims_in_dir_names.pop(0)
            image_path = ims_in_dir.pop(0)
            idx = np.where(self.labels.image_names==image_name)[0]

            images.append(self._load_image(image_path))
            labels.append(self.labels.numerical[idx])
            image_names.append(image_name)

            if len(image_names) == self.batch_size:
                labels = np.array(labels)
                yield np.array(images), np.reshape(labels, (len(labels))), image_names
                images = list()
                labels = list()
                image_names = list()

        if len(image_names):
            labels = np.array(labels)
            yield np.array(images), np.reshape(labels, (len(labels))), image_names

    def _load_image(self, path):
        grayscale = self.channels == 1
        return tools.load_image(path, (self.target_imsize[1], self.target_imsize[0]), grayscale)