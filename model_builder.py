from os import cpu_count
import logging

from keras.optimizers import Adadelta, SGD, Adam, Adagrad
import tensorflow as tf
import keras

import tools
from accumulated_adadelta_optimizer import AccumAdadelta
import dnns


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if tf.test.is_gpu_available():
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': cpu_count()})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


class ModelBuilder:

    KNOWN_MODELS = ['scenenet_v1']

    KNOWN_MODES = tools.MODES

    CUST_OBJS = {'AccumAdadelta': AccumAdadelta}

    OPTIMIZERS = {'Adadelta': Adadelta,
                  'SGD': SGD,
                  'Adam': Adam,
                  'Adagrad': Adagrad,
                  'AccumAdadelta': AccumAdadelta,
                  }

    def __init__(self, configs, mode=None, model_name=None, model_path=None, classes=None, loss=None, optimizer=None):
        self.configs = configs
        self.log = logging.getLogger('root')
        self.mode = mode
        self.model_name = model_name or configs['model_name']
        self.classes = classes or configs['classes']
        self.loss = loss or configs['loss']
        self.model_path = tools.str2path(model_path or configs['model_path'])
        self.optimizer_type = optimizer or configs['optimizer']
        self.metrices_names = configs.get('metrices', [])
        self.custom_objects = list()

        self._validate_input()
        self._add_custom_optimizer()
        self._add_custom_metrices()
        self._add_custom_loss()

        self._optimizer = self._init_optimizer()
        self._metrices = self._select_metrices()

    def _add_custom_optimizer(self):
        if self.optimizer_type in self.CUST_OBJS:
            self.custom_objects.append(self.optimizer_type)

    def _add_custom_metrices(self):
        for metric in self.metrices_names:
            if metric in self.CUST_OBJS:
                self.custom_objects.append(metric)

    def _add_custom_loss(self):
        if self.loss in self.CUST_OBJS:
            self.custom_objects.append(self.loss)

    def _validate_input(self):
        if self.mode not in self.KNOWN_MODES:
            raise IOError(f"Uknown mode {self.mode}, known are: {self.KNOWN_MODES}")

        if self.mode != 'train' and not self.model_path.exists():
            raise IOError(f'{self.model_path} does not exist')

        if self.model_name not in self.KNOWN_MODELS:
            raise IOError(f"Uknown model {self.model_name}, known are: {self.KNOWN_MODELS}")

    def _select_model(self):
        self.log.info(f'Network selected: {self.model_name}')
        model = None
        if self.model_name == 'scenenet_v1':
            net = dnns.scenenet_v1
            model = net(**self.configs.get('network_parameters', {}))

        return model

    def _init_optimizer(self):
        if self.optimizer_type in self.CUST_OBJS:
            opt = self.CUST_OBJS[self.optimizer_type](**self.configs.get('optimizer_parameters', {}))
        else:
            opt = self.OPTIMIZERS[self.optimizer_type](**self.configs.get('optimizer_parameters', {}))
        return opt

    def _custom_objects(self):
        cobjs = dict()
        required_cojbs = self.custom_objects
        for robj in required_cojbs:
            if robj in self.CUST_OBJS.keys():
                cobjs.update({robj: self.CUST_OBJS[robj]})
        return cobjs

    def _select_metrices(self):
        metrices = list()
        for metric in self.metrices_names:
            if metric in self.CUST_OBJS:
                metrices.append(self.CUST_OBJS[metric])
        return metrices

    def build(self):
        model = None
        if self.mode == 'train':
            model = self._select_model()

            model.compile(optimizer=self._optimizer, loss=self.loss, metrics=self._metrices)
            self.log.info(f"Model created and compiled. Loss: {self.loss}, "
                          f"optimizer: {self._optimizer}, metrics: {self._metrices}")

        elif self.mode != 'train':
            cust_objs = self._custom_objects()
            model = tools.load_keras_model(self.model_path, custom_objects=cust_objs)

            self.log.info(f"Model loaded from {self.model_path}")

        model.summary()
        return model
