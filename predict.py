import numpy as np

from model_builder import ModelBuilder
from data_generator import DataGenerator
import tools

FDIR = tools.str2path(__file__).parent


def predict(config_path, model_path=None, labels_json=None, data_dir=None, group=None):
    configs = tools.read_configs(config_path)

    model_path = tools.str2path(model_path or configs.get('model_path'))
    data_dir = tools.str2path(data_dir or configs.get('data_dir'))
    labels_json = tools.str2path(labels_json or configs.get('labels_json'))
    group = group or configs.get('group')

    model_builder = ModelBuilder(configs, mode='predict', model_path=model_path.as_posix())
    model = model_builder.build()

    pred_gen = DataGenerator(configs, image_dir=data_dir, labels_json=labels_json, group=group, mode='predict')

    correct = 0
    false = 0
    for x, y, xnames in pred_gen.flow_from_labels():
        predictions = model.predict(x, verbose=1)
        pred_cls_ids = np.argmax(predictions, axis=1)
        tp = np.sum(pred_cls_ids==y)
        error = len(y) - tp
        correct += tp
        false += error
    accuracy = (1 - false / correct) * 100

    print(f"Correct: {correct}")
    print(f'False: {false}')
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument('--cfg',
                      required=True)
    argp.add_argument('--im-dir',
                      dest='im_dir',
                      default=None)
    argp.add_argument('--model',
                      default=None)
    argp.add_argument('--group',
                      default=None)
    argp.add_argument('--labels',
                      dest='labels',
                      default=None)

    args = argp.parse_args()
    tools.basic_logger(log_path=(FDIR / 'logs'))

    predict(args.cfg, args.model, args.labels, args.im_dir, args.group)
