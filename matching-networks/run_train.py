import argparse
import configparser
import time, os
import datetime
from shutil import copyfile

import numpy as np
import tensorflow as tf
from model import MatchingNetwork
from load_data import load


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.train_support',
                  'data.test_support', 'data.train_query', 'data.test_query',
                  'data.query', 'data.support', 'data.way', 'data.episodes',
                  'model.lstm_size', 'train.epochs', 'train.patience', 'data.batch', 'train.restore']
    float_params = ['train.lr']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict

def train(config):

    # Create folder for model
    model_dir = config['model.save_dir'][:config['model.save_dir'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # We want 'trainval' for omniglot
    splitting = ['train', 'trainval'][config['data.dataset'] == 'omniglot']
    data_dir = config['data.dataset_path']
    ret = load(data_dir, config, [splitting, 'val'])
    train_loader = ret[splitting]
    val_loader = ret['val']

    # Setup training operations
    way = config['data.train_way']
    lstm_dim = config['model.lstm_size']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = MatchingNetwork(way=way, w=w, h=h, c=c, lstm_size=lstm_dim)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    def run_optimization(x_support, y_support, x_query, y_query):   # train_step
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(x_support, y_support, x_query, y_query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for epoch in range(config['train.epochs']):
        for i_episode in range(config['data.episodes']):
            x_support, y_support, x_query, y_query = train_loader.get_next_episode()
            run_optimization(x_support, y_support, x_query, y_query)
            loss, acc = model(x_support, y_support, x_query, y_query)

            if i_episode % 5 == 0:
                loss, acc = model(x_support, y_support, x_query, y_query)
                print("epoch: %i, episode: %i, loss: %f, acc: %f" %(epoch, i_episode, np.mean(loss.numpy()), acc.numpy() * 100))

    print("Training succeed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, default="config_omniglot.conf", help="Path to the config file.")

    time_start = time.time()

    # Run training
    args = vars(parser.parse_args())
    config = configparser.ConfigParser()
    config.read(args['config'])
    config = preprocess_config(config['TRAIN'])
    train(config)

    time_end = time.time()

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60
    print(f"Training took: {h} h {min} min {sec} sec")
