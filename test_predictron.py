# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:02:20 2020

@author: RTS
"""

from predictron import Predictron
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + "C:\\Users\\rts\\Anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin\\graphviz"

class Arg_parser():
    def __init__(self):
        self.train_dir = './ckpts/predictron_train'
        self.max_steps =  10
        self.num_gpus = 1
        
        # adam optimizer:
        self.learning_rate = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.epochs = 5000
        self.batch_size = 32
        self.state_size = 1000
        self.max_depth = 4
        self.max_grad_norm = 10.
        self.log_device_placement = False
        self.num_threads = 10
        
        # adam optimizer:
        self.learning_rate = 1e-2
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-7
        
        self.l2_weight=0.01
        self.dropout_rate=0.
        
        # self.epochs = 5000
        self.batch_size = 5
        self.episode_length = 2
        self.predictron_update_rate = 500
        self.burnin = 0*1e4
        self.gamma = 0.99
        self.replay_memory_size = 100000
        self.predictron_update_steps = 50
        self.max_depth = 16
        
        self.DQN_train_steps = 5e4
        self.DQN_train_steps_initial = 5e4
        self.Predictron_train_steps = 5e4
        self.Predictron_train_steps_initial = 5e4
        self.train_itterations = 10
        
        
        self.state_rep_size = 128#args.state_rep_size

        self.seed = 0#args.seed
    
config = Arg_parser()

predictron = Predictron(config)
model = predictron.model
data = np.random.rand(config.batch_size,config.state_size)
target = np.random.rand(config.batch_size,1)
# target = np.concatenate((target, target),axis=1)
# print(target.shape)

preturn_arr = []
lambda_preturn_arr = []
for i in range(config.epochs):
    data = np.random.rand(config.batch_size,config.state_size)
    target = np.random.rand(config.batch_size,1)
    _, preturn, lambda_preturn = model.train_on_batch(data, target)
    print(preturn,lambda_preturn)
    preturn_arr.append(preturn)
    lambda_preturn_arr.append(lambda_preturn)
    if i % 10 == 0:
        print(str(100*(i+1)/config.epochs) + '% done')

plt.figure(1)
plt.plot(preturn_arr)
plt.figure(2)
plt.plot(lambda_preturn_arr)


test_data = np.random.rand(1,config.state_size)
print(model(test_data))