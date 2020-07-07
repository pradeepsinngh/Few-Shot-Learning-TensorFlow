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


def train(config):

    # Create folder for model
    model_dir = config['model.save_path'][:config['model.save_path'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load data
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Setup training operations
    n_support = config['data.train_support']
    n_query = config['data.train_query']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    model = Prototypical(n_support, n_query, w, h, c)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    def run_optimization(support, query):   # train_step
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for epoch in range(config['train.epochs']):
        for i_episode in range(config['data.episodes']):
            support, query = train_loader.get_next_episode()
            run_optimization(support, query)

            if i_episode % 5 == 0:
                loss, acc = model(support, query)
                print("epoch: %i, episode: %i, loss: %f, acc: %f" %(epoch, i_episode, loss, acc * 100))

    print("Training succeed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, default="config_omniglot.conf",
                    help="Path to the config file.")

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
