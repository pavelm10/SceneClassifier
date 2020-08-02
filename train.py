import logging
import numpy as np

from tensorflow import keras

import tools
from data_generator import DataGenerator
from model_builder import ModelBuilder


FDIR = tools.str2path(__file__).parent


def train(config_path, train_dir=None, val_dir=None, output_dir=None, train_labels_json=None, val_labels_json=None,
          group=None, model_name=None, model_suffix=None):
    np.random.seed(42)  # for reproducibility
    logger = logging.getLogger('root')
    configs = tools.read_configs(config_path)

    train_dir = tools.str2path(train_dir or configs['train_dir'])
    val_dir = tools.str2path(val_dir or configs['val_dir'])
    train_labels_json = tools.str2path(train_labels_json or configs['train_labels_json'])
    val_labels_json = tools.str2path(val_labels_json or configs['val_labels_json'])
    output_dir = tools.str2path(output_dir or configs['output_dir'])
    group = group or configs['group']
    model_name = model_name or configs['model_name']
    model_suffix = model_suffix or configs['model_suffix']

    output_dir.mkdir(exist_ok=True)

    model_out_name = f'{model_name}_{group}_{model_suffix}.h5'
    model_path = output_dir / model_out_name

    train_gen = DataGenerator(configs, train_dir, train_labels_json, 'train', group)
    val_gen = DataGenerator(configs, val_dir, val_labels_json, 'val', group)

    epochs = configs['epochs']
    classes = configs['network_parameters']['classes']
    loss = configs['loss']
    optimizer = configs['optimizer']

    model_builder = ModelBuilder(configs, 'train', model_name, model_path, classes, loss, optimizer)
    model = model_builder.build()

    checkpoint = keras.callbacks.ModelCheckpoint(model_path.as_posix(), monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    logger.info(f"Training model {model_out_name} for {epochs} epochs")
    logger.info(f'Class weights: {train_gen.class_weights}')

    model.fit_generator(generator=train_gen.flow_generator,
                        steps_per_epoch=train_gen.steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        class_weight=train_gen.class_weights,
                        callbacks=[checkpoint],
                        validation_data=val_gen.flow_generator,
                        validation_steps=val_gen.steps_per_epoch)


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument('--cfg',
                      required=True)
    argp.add_argument('--train-dir',
                      dest='train_dir',
                      default=None)
    argp.add_argument('--val-dir',
                      dest='val_dir',
                      default=None)
    argp.add_argument('--train-labels',
                      dest='train_labels_json',
                      default=None)
    argp.add_argument('--val-labels',
                      dest='val_labels_json',
                      default=None)
    argp.add_argument('--output',
                      dest='output_dir',
                      default=None)
    argp.add_argument('--group',
                      default=None)
    argp.add_argument('--model-name',
                      dest='model_name',
                      default=None)
    argp.add_argument('--model-suffix',
                      dest='model_suffix',
                      default=None)

    args = argp.parse_args()
    tools.basic_logger(log_path=(FDIR / 'logs'))

    train(args.cfg, args.train_dir, args.val_dir, args.output_dir, args.train_labels_json, args.val_labels_json,
          args.group, args.model_name, args.model_suffix)
