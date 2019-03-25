import os
import re
import numpy as np
from subprocess import check_output
from .features import extract_features

class Session:
    """
    A class to represent a logic synthesis optimization session using ABC
    """
    def __init__(self, params):
        self.params = params

        self.action_space_length = len(self.params['optimizations'])
        self.observation_space_size = 9     # number of features

        self.iteration = 0
        self.episode = 0
        self.sequence = ['strash']
        self.delay, self.area = float('inf'), float('inf')
    
    def reset(self):
        """
        resets the environment and returns the state
        """
        self.iteration = 0
        self.episode += 1
        self.delay, self.area = float('inf'), float('inf')
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.params['playground_dir'], str(self.episode))
        if not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)

        state, _ = self._run()
        return state
    
    def step(self, optimization):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][optimization])
        new_state, reward = self._run()

        return new_state, reward, self.iteration == self.params['iterations'], None

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.blif')
    
        abc_command = 'read ' + self.params['mapping']['library_file'] + '; '
        abc_command += 'read ' + self.params['design_file'] + '; '
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'map -D ' + str(self.params['mapping']['clock_period']) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'print_stats; '
    
        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            # get reward
            delay, area = self._get_metrics(proc)
            reward = self._get_reward(delay, area)
            self.delay, self.area = delay, area
            # get new state of the circuit
            state = self._get_state(output_design_file)
            return state, reward
        except Exception as e:
            print(e)
            return None, None
        
    def _get_metrics(self, stats):
        """
        parse delay and area from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        
        ob = re.search(r'delay *= *[0-9]+.?[0-9]+', line)
        delay = float(ob.group().split('=')[1].strip())
        
        ob = re.search(r'area *= *[0-9]+.?[0-9]+', line)
        area = float(ob.group().split('=')[1].strip())

        return delay, area
    
    def _get_reward(self, delay, area):
        constraint_met = True
        optimization_improvement = 0    # (-1, 0, 1) <=> (worse, same, improvement)

        # check optimizing parameter
        if self.params['optimize_for'] == 'delay':
            if delay < self.delay:
                optimization_improvement = 1
            elif delay == self.delay:
                optimization_improvement = 0
            else:
                optimization_improvement = -1
        elif self.params['optimize_for'] == 'area':
            if area < self.area:
                optimization_improvement = 1
            elif delay == self.delay:
                optimization_improvement = 0
            else:
                optimization_improvement = -1
        else:
            raise Exception('Optimizing parameter not set correctly!')
        
        # check constraint parameter
        if self.params['constraint_for'] is not None:
            if self.params['constraint_for'] == 'delay':
                if delay > self.params['constraint_value']:
                    constraint_met = False
            elif self.params['constraint_for'] == 'area':
                if area > self.params['constraint_value']:
                    constraint_met = False

        # now calculate the reward
        return self._reward_table(constraint_met, optimization_improvement)
    
    def _reward_table(self, constraint_met, optimization_improvement):
        return {
            True: {
                1: 2,
                0: 0,
                -1: -1
            },
            False: {
                1: 1 if self.iteration < self.params['iterations'] / 2 else -1,
                0: -1,
                -1: -2
            }
        }[constraint_met][optimization_improvement]
    
    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])
    