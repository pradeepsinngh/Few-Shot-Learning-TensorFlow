import argparse
import configparser
import os, time, datetime
import numpy as np
import tensorflow as tf
from load_data import load
from model import SiameseNet

def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.batch',
                  'data.episodes', 'train.epochs', 'train.patience']
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

    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Setup training operations
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = SiameseNet(w, h, c)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    def run_optimization(support, query, labels):   # train_step
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for epoch in range(config['train.epochs']):
        for i_episode in range(config['data.episodes']):
            support, query, labels = train_loader.get_next_episode()
            run_optimization(support, query, labels)
            #loss, acc = model(support, query, labels)

            if i_episode % 5 == 0:
                loss, acc = model(support, query, labels)
                print(loss, acc)
                print("epoch: %i, episode: %i, loss: %f, acc: %f" %(epoch, i_episode, np.mean(loss.numpy()), acc.numpy() * 100))

    print("Training succeed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, default="omniglot.conf", help="Path to the config file.")

    # Run training
    args = vars(parser.parse_args())
    config = configparser.ConfigParser()
    config.read(args['config'])
    config = preprocess_config(config['TRAIN'])
    train(config)
