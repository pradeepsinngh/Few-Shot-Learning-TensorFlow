import tensorflow as tf
import argparse
import configparser
import os
import time
import numpy as np

from model import MatchingNetwork
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
    model_dir = config['model.save_dir'][:config['model.save_dir'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load data
    data_dir = config['data.dataset_path']
    ret = load(data_dir, config, ['test'])
    test_loader = ret['test']

    # Setup validation operations
    way = config['data.test_way']
    lstm_dim = config['model.lstm_size']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    model = MatchingNetwork(way, w, h, c, lstm_size=lstm_dim)
    model.load(config['model.save_dir'])

    def calc_loss(x_support, y_support, x_query, y_query):
        loss, acc = model(x_support, y_support, x_query, y_query)
        return loss, acc

    for i_episode in tqdm(range(config['data.episodes'])):
        x_support, y_support, x_query, y_query = test_loader.get_next_episode()
        if i_episode % 5 == 0:
            print("episode: %i" %(i_episode))
        loss, acc = calc_loss(x_support, y_support, x_query, y_query)
        print("Loss: %f, Accuracy: %f" %(loss.numpy(), acc.numpy() * 100))

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
