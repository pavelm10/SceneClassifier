import pathlib
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import cv2
import shutil
import ruamel.yaml


MODES = ['train', 'val', 'test', 'predict', 'retrain']
GROUPS = ['scene', 'weather', 'timeofday']


def str2path(path_str):
    return pathlib.Path(path_str) if path_str is not None else None


def store_json(json_path, data):
    with open(json_path, 'w') as json_handle:
        json.dump(data, json_handle)


def load_json(json_path):
    with open(json_path, 'r') as json_handle:
        data = json.load(json_handle)
        return data


def extract_labels(input_json, output_json=None, blacklist=None, mapping=None):
    data = load_json(input_json)
    extracted_labels = dict()
    map_keys = set(list(mapping.keys()))
    blacklist = set(blacklist)
    for label in data:
        label_data = list(label['attributes'].values())
        if not blacklist.intersection(label_data):
            extracted_labels[label['name']] = label['attributes']
        else:
            continue

        intersec = list(map_keys.intersection(label_data))
        for key in intersec:
            try:
                idx = label_data.index(key)
                label_data[idx] = mapping[key]
            except ValueError:
                continue

        if intersec:
            attr_keys = list(label['attributes'].keys())
            new_dict = dict().fromkeys(attr_keys, label_data)
            extracted_labels[label['name']] = new_dict

    if output_json:
        store_json(output_json, extracted_labels)

    return extracted_labels


def get_attribute_stats(attribute_list):
    unique = list(set(attribute_list))
    stats = dict()
    attribute_array = np.array(attribute_list)
    for unique_val in unique:
        stats[unique_val] = int(np.sum(attribute_array == unique_val))
    return stats


def generate_statistics(input_json, output_json=None):
    data = load_json(input_json)
    attr_data = dict().fromkeys(GROUPS, [])
    for attributes in data.values():
        for attr in GROUPS:
            attr_data[attr].append(attributes[attr])

    stats = dict()
    for attr in GROUPS:
        stats[attr] = get_attribute_stats(attr_data[attr])

    if output_json:
        store_json(output_json, stats)
    return stats


def get_images_meta(input_json, image_dir, output_json):
    rgb_bw_dtype = [('r', 'f8'), ('g', 'f8'), ('b', 'f8'), ('bw', 'f8')]
    meta = load_json(input_json)
    out_data = dict()
    means = np.zeros((len(meta), ), dtype=rgb_bw_dtype)
    varis = np.zeros((len(meta), ), dtype=rgb_bw_dtype)
    for idx, name in enumerate(meta.keys()):
        impath = str2path(image_dir) / name
        bgr_image = cv2.imread(impath.as_posix(), 1)
        bw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        means['bw'][idx] = np.mean(bw_image)
        means['r'][idx] = np.mean(bgr_image[:, :, 2])
        means['g'][idx] = np.mean(bgr_image[:, :, 1])
        means['b'][idx] = np.mean(bgr_image[:, :, 0])
        varis['bw'][idx] = np.var(bw_image)
        varis['r'][idx] = np.var(bgr_image[:, :, 2])
        varis['g'][idx] = np.var(bgr_image[:, :, 1])
        varis['b'][idx] = np.var(bgr_image[:, :, 0])

    out_data['mu_bw'] = np.mean(means['bw'])
    out_data['mu_r'] = np.mean(means['r'])
    out_data['mu_g'] = np.mean(means['g'])
    out_data['mu_b'] = np.mean(means['b'])
    out_data['std_bw'] = np.sqrt(np.mean(varis['bw']))
    out_data['std_r'] = np.sqrt(np.mean(varis['r']))
    out_data['std_g'] = np.sqrt(np.mean(varis['g']))
    out_data['std_b'] = np.sqrt(np.mean(varis['b']))

    store_json(output_json, out_data)


def bgr2rgb(image):
    """
    converts bgr image to rgb format
    :param image: np.array
    :return: np.array
    """
    return image[..., ::-1]


def basic_logger(name='root', log_path=None, file_level=logging.INFO, stream_level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_hanlder = logging.StreamHandler()
    stream_hanlder.setFormatter(formatter)
    stream_hanlder.setLevel(stream_level)
    log.addHandler(stream_hanlder)

    if log_path:
        log_path.mkdir()
        logfile = log_path / f"{name}.log"
        file_handler = RotatingFileHandler(logfile, maxBytes=5242880, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        log.addHandler(file_handler)

    return log


def create_keras_image_directory_tree(image_dir=None, output_dir=None, labels_json=None, group=None):
    image_dir = str2path(image_dir)
    output_dir = str2path(output_dir)
    labels = load_json(labels_json)
    subdirs = list(set([label[group] for label in labels.values()]))

    for subdir in subdirs:
        subdir_path = output_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

    for name, label in labels.items():
        im_src_path = image_dir / name
        im_dest_path = output_dir / label[group] / name
        shutil.copy(im_src_path.as_posix(), im_dest_path.as_posix())


def read_configs(cfg_path):
    with open(cfg_path) as stream:
        configs = ruamel.yaml.safe_load(stream)
        return configs


def load_keras_model(model_path, custom_objects=None):
    from keras.models import load_model
    if custom_objects:
        return load_model(model_path, custom_objects=custom_objects)
    else:
        return load_model(model_path)
