#!/usr/bin/python3
import yaml
import os
import argparse
import datetime
import numpy as np
from drills.model import A2C
from drills.log import log
from pyfiglet import Figlet

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
    parser.add_argument('-v', '--version', action='version', version = 'CNV-Sim v0.1', help = "Shows program's version number and exit")
    parser.add_argument("mode", type=str, choices=['train', 'optimize'], \
        help="Use the design to train the model or only optimize it")
    parser.add_argument("params", type=open, nargs='?', default='params.yml', \
        help="Path to the params.yml file")
    args = parser.parse_args()
    
    options = yaml.load(args.params, Loader=yaml.FullLoader)

    f = Figlet(font='slant')
    print(f.renderText('DRiLLS'))
    
    if args.mode == 'train':
        log('Starting to train the agent ..')
        
        all_rewards = []
        learner = A2C(options)
        for i in range(options['episodes']):
            log('Episode: ' + str(i+1))
            total_reward = learner.train_episode()
            all_rewards.append(total_reward)
            log('Episode: ' + str(i) + ' - done with total reward = ' + str(total_reward))
            print('')
    
        mean_reward = np.mean(all_rewards[-100:])
