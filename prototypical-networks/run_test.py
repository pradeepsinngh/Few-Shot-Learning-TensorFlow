import unittest
import tensorflow as tf
import argparse
import configparser
import os
import time
import numpy as np

from model import Prototypical
from load_data import load

def test(config):

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
    print('Model -- ', model_path)
    model.load(model_path)
    print("Model loaded.")

    for i_episode in range(config['data.episodes']):
        support, query = val_loader.get_next_episode()
        if i_episode % 5 == 0:
            print("episode: %i" %(i_episode))
        loss, acc = model(support, query)
        print("episode: %i, loss: %f, acc: %f" %(i_episode, loss, acc * 100))

    print("Testing Done!")


if __name__ == '__main__':

    test_1_shot_5_way = {
            "data.dataset": "omniglot",
            "data.split": "vinyals",
            "data.train_way": 5,
            "data.train_support": 1,
            "data.train_query": 1,
            "data.test_way": 5,
            "data.test_support": 1,
            "data.test_query": 1,
            "data.episodes": 10,
            "model.x_dim": "28,28,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": './results/models/omniglot1234_trainval.h5'}

    test_5_shot_5_way = {
            "data.dataset": "omniglot",
            "data.split": "vinyals",
            "data.train_way": 5,
            "data.train_support": 5,
            "data.train_query": 5,
            "data.test_way": 5,
            "data.test_support": 5,
            "data.test_query": 5,
            "data.episodes": 10,
            "model.x_dim": "28,28,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": './results/models/omniglot1234_trainval.h5'}

    test_10_shot_1_way = {
            "data.dataset": "omniglot",
            "data.split": "vinyals",
            "data.train_way": 1,
            "data.train_support": 10,
            "data.train_query": 10,
            "data.test_way": 1,
            "data.test_support": 10,
            "data.test_query": 10,
            "data.episodes": 10,
            "model.x_dim": "28,28,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": './results/models/omniglot1234_trainval.h5'}

    test_1_shot_50_way = {
            "data.dataset": "omniglot",
            "data.split": "vinyals",
            "data.train_way": 50,
            "data.train_support": 1,
            "data.train_query": 1,
            "data.test_way": 50,
            "data.test_support": 1,
            "data.test_query": 1,
            "data.episodes": 10,
            "model.x_dim": "28,28,1",
            "model.z_dim": 64,
            "train.epochs": 2,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": './results/models/omniglot1234_trainval.h5'}

    print("Test -- test_1_shot_5_way")
    test(test_1_shot_5_way)
    print("Test -- test_5_shot_5_way")
    test(test_5_shot_5_way)
    print("Test -- test_10_shot_1_way")
    test(test_10_shot_1_way)
    print("Test -- test_1_shot_50_way")
    test(test_1_shot_50_way)
