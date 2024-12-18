import numpy as np
import os, sys
import pickle
import yaml
from easydict import EasyDict as edict
import re
from typing import Any, IO
import json
import glob

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")
    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

def get_data(data_path):
    classes = { 'normal' : 0, 'model': 1 }
    all_json_paths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(data_path, cls, 'json')
        json_path_list = glob.glob(os.path.join(cls_dir, '*.json'))

        all_json_paths.extend(json_path_list)
        labels.extend([classes[cls] for _ in range(len(json_path_list))])

    return all_json_paths, labels

def split_dataset_labels_kcv(all_json_paths, labels, k, i):
    """
    Split the dataset into k folds and return the i-th fold
    """
    n = len(all_json_paths)
    assert n == len(labels)
    assert i < k

    train_json_paths, train_labels, test_json_paths, test_labels = [], [], [], []
    for j in range(n):
        if j % k == i:
            test_json_paths.append(all_json_paths[j])
            test_labels.append(labels[j])
        else:
            train_json_paths.append(all_json_paths[j])
            train_labels.append(labels[j])

    return train_json_paths, train_labels, test_json_paths, test_labels

def display_train_test_results(save_path, i, all_accs_train, all_loss_train, all_accs_test, all_loss_test):
    """
    Display the training and testing results
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(all_accs_train, label='train acc')
    plt.plot(all_accs_test, label='test acc')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'acc_{i}.png'))

    plt.figure()
    plt.plot(all_loss_train, label='train loss')
    plt.plot(all_loss_test, label='test loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'loss_{i}.png'))

    plt.close()

def print_kcv_results(kcv_results):
    """
    Print the results of kcv
    """
    for i in range(len(kcv_results)):
        print(f"Fold {i}, last 5 epochs:")
        # print the last 5 epochs
        print(f"Train acc: {np.mean(kcv_results[i]['train_accs'][-5:]):.4f}")
        print(f"Test acc: {np.mean(kcv_results[i]['test_accs'][-5:]):.4f}")
        print(f"Train loss: {np.mean(kcv_results[i]['train_losses'][-5:]):.4f}")
        print(f"Test loss: {np.mean(kcv_results[i]['test_losses'][-5:]):.4f}")
        print()


def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)  # Remove special characters
    caption = re.sub(r'\s+', ' ', caption).strip()  # Normalize whitespace
    caption = re.sub(r'\d+', '', caption)  # Remove numbers
    return caption


def get_json_paths_and_caption(data_path):
    """
    Get the json paths and captions from data_path/model/ and data_path/normal/
    """
    classes = ['normal', 'model']
    json_id2path = {} # normal_1: path
    captions = []
    json_paths = []

    # collect all json paths
    for cls in classes:
        cls_dir = os.path.join(data_path, cls, 'json')
        json_path_list = glob.glob(os.path.join(cls_dir, '*.json'))
        for json_path in json_path_list:
            json_id = os.path.splitext(os.path.basename(json_path))[0]
            json_id2path[cls + '_' + json_id] = json_path

    for cls in classes:
        cls_dir = os.path.join(data_path, cls)
        caption_path_list = glob.glob(os.path.join(cls_dir, 'annot', '*.txt'))
        for caption_path in caption_path_list:
            json_id_caption_id = os.path.splitext(os.path.basename(caption_path))[0]
            # if caption_id != 6 continue
            if json_id_caption_id.split('_')[1] != '6':
                continue
            json_id = cls + '_' + json_id_caption_id.split('_')[0]
            if json_id in json_id2path:
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
                    captions.append(clean_caption(caption))
                    json_paths.append(json_id2path[json_id])

    assert len(json_paths) == len(captions)

    return json_paths, captions

def save_vocab(vocab, vocab_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    return
