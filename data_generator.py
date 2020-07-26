import logging

from keras.preprocessing.image import ImageDataGenerator

import tools


class DataGenerator:

    MODES = tools.MODES
    GROUPS = tools.GROUPS

    def __init__(self, config, image_dir=None, labels_json=None, mode=None, group=None):
        self.log = logging.getLogger('root')
        self.config = config
        self.image_dir = tools.str2path(image_dir)
        self.labels_json = tools.str2path(labels_json)
        self.mode = mode
        self.group = group
        self._validate_input()

        self.target_imsize = (self.config['target_height'], self.config['target_width'])
        self.keras_dir = self.image_dir / self.group
        self.keras_dir.mkdir()
        self.labels_data = tools.load_json(labels_json)
        self.log.info(f'Creating Keras Directory Tree - {self.mode}...')
        tools.create_keras_image_directory_tree(self.image_dir, self.keras_dir, self.labels_json, self.group)
        self._data_generator = self._init_data_gen()

    @property
    def data_generator(self):
        return self._data_generator

    @property
    def samples(self):
        return len(self.labels_data)

    def _validate_input(self):
        if not self.image_dir.exists():
            raise IOError(f'{self.image_dir} does not exist')

        if not self.labels_json.exists():
            raise IOError(f'{self.labels_json} does not exist')

        if self.mode not in self.MODES:
            raise AttributeError(f'{self.mode} is unknown mode, known are {self.MODES}')

        if self.group not in self.GROUPS:
            raise AttributeError(f'{self.group} is unknown group, known are {self.GROUPS}')

    def _init_data_gen(self):
        cfg = self.config.get(f'{self.mode}_gen', {})
        return ImageDataGenerator(**cfg)

    def flow_from_directory(self):
        cfg = self.config.get(f'{self.mode}_flow', {})
        return self._data_generator.flow_from_directory(self.keras_dir, target_size=self.target_imsize, **cfg)
