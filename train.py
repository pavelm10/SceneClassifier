import logging
import numpy as np

from keras.callbacks import ModelCheckpoint

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

    model_out_name = f'{model_name}_{group}_{model_suffix}.h5'
    model_path = output_dir / model_out_name

    train_gen = DataGenerator(configs, train_dir, train_labels_json, 'train', group)
    val_gen = DataGenerator(configs, val_dir, val_labels_json, 'val', group)

    nb_train = configs.get('train_samples', None) or train_gen.samples
    nb_val = configs.get('val_samples', None) or val_gen.samples

    epochs = configs['epochs']

    model_builder = ModelBuilder(configs, logger)
    model = model_builder.build()

    checkpoint = ModelCheckpoint(model_path.as_posix(), monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    logger.info(f"Training model {model_out_name} for {epochs} epochs")

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=nb_train,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=val_gen,
                        validation_steps=nb_val)


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
