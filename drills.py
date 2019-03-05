#!/usr/bin/python3
import yaml
import os
import numpy as np
from drills import A2C

if __name__ == '__main__':
    data_file = 'data.yml'

    with open(data_file, 'r') as f:
        options = yaml.load(f)

    all_rewards = []
    
    learner = A2C(options)
    for i in range(1000):
        total_reward = learner.train_episode(i)
        all_rewards.append(total_reward)
    
    mean_reward = np.mean(all_rewards[-100:])
