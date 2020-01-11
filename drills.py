#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import yaml
import os
import argparse
import datetime
import numpy as np
import time
from drills.model import A2C
from drills.fixed_optimization import optimize_with_fixed_script
from pyfiglet import Figlet

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)

class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter, \
        description='Performs logic synthesis optimization using RL')
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version', \
        version = 'DRiLLS v0.1', help="Shows program's version number and exit")
    parser.add_argument("-l", "--load_model", action='store_true', \
        help="Loads a saved Tensorflow model")
    parser.add_argument("-s", "--fixed_script", type=open, \
        help="Executes a fixed optimization script before DRiLLS")
    parser.add_argument("mode", type=str, choices=['train', 'optimize'], \
        help="Use the design to train the model or only optimize it")
    parser.add_argument("mapping", type=str, choices=['scl', 'fpga'], \
        help="Map to standard cell library or FPGA")
    parser.add_argument("params", type=open, nargs='?', default='params.yml', \
        help="Path to the params.yml file")
    args = parser.parse_args()
    
    options = yaml.load(args.params, Loader=yaml.FullLoader)

    f = Figlet(font='slant')
    print(f.renderText('DRiLLS'))

    if args.fixed_script:
        params = optimize_with_fixed_script(params, args.fixed_script)

    if args.mapping == 'scl':
        fpga_mapping = False
    else:
        fpga_mapping = True
    
    if args.mode == 'train':
        log('Starting to train the agent ..')
        
        all_rewards = []
        learner = A2C(options, load_model=args.load_model, fpga_mapping=fpga_mapping)
        training_start_time = time.time()
        for i in range(options['episodes']):
            log('Episode: ' + str(i+1))
            start = time.time()
            total_reward = learner.train_episode()
            end = time.time()
            all_rewards.append(total_reward)
            log('Episode: ' + str(i) + ' - done with total reward = ' + str(total_reward))
            log('Episode ' + str(i) + ' Run Time ~ ' + str((start - end) / 60) + ' minutes.')
            print('')
        training_end_time = time.time()
        log('Total Training Run Time ~ ' + str((training_end_time - training_start_time) / 60) + ' minutes.')
    
        mean_reward = np.mean(all_rewards[-100:])
    elif args.mode == 'optimize':
        log('Starting agent to optimize')
        learner = A2C(options, load_model=True)
        for _ in range(options['iterations']):
            # TODO: iteratively run the optimizer
            pass