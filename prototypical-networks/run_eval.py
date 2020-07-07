import tensorflow as tf
import argparse
import configparser
import os
import time
import numpy as np

from model import Prototypical
from load_data import load

def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.train_support',
                      'data.test_support', 'data.train_query', 'data.test_query',
                      'data.query', 'data.support', 'data.way', 'data.episodes',
                      'model.z_dim', 'train.epochs',
                      'train.patience']

    float_params = ['train.lr']

    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


def eval(config):

    # Create folder for model
    model_dir = config['model.save_path'][:config['model.save_path'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load data
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['val'])
    val_loader = ret['val']

    # Setup validation operations
    n_support = config['data.test_support']
    n_query = config['data.test_query']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    model = Prototypical(n_support, n_query, w, h, c)
    model_path = f"{config['model.save_path']}"
    model.load(model_path)
    print("Model loaded.")

    for i_episode in range(config['data.episodes']):
        support, query = val_loader.get_next_episode()
        if i_episode % 5 == 0:
            print("episode: %i" %(i_episode))
        loss, acc = model(support, query)
        print("episode: %i, loss: %f, acc: %f" %(i_episode, loss, acc * 100))

    print("Evaluation Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Evaluation')
    parser.add_argument("--config", type=str, default="config_omniglot.conf",
                    help="Path to the config file.")

    # Run training
    args = vars(parser.parse_args())
    config = configparser.ConfigParser()
    config.read(args['config'])
    config = preprocess_config(config['EVAL'])
    eval(config)
