import os
import re
import datetime
import numpy as np
from subprocess import check_output
from .features import extract_features

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)

class SCLSession:
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

        self.best_known_area = (float('inf'), float('inf'), -1, -1)
        self.best_known_delay = (float('inf'), float('inf'), -1, -1)
        self.best_known_area_meets_constraint = (float('inf'), float('inf'), -1, -1)

        # logging
        self.log = None
    
    def __del__(self):
        if self.log:
            self.log.close()
    
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
        
        # logging
        log_file = os.path.join(self.episode_dir, 'log.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, area, delay, best_area_meets_constraint, best_area, best_delay\n')

        state, _ = self._run()

        # logging
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(self.area), str(self.delay)]) + '\n')
        self.log.flush()

        return state
    
    def step(self, optimization):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][optimization])
        new_state, reward = self._run()

        # logging
        if self.area < self.best_known_area[0]:
            self.best_known_area = (self.area, self.delay, self.episode, self.iteration)
        if self.delay < self.best_known_delay[1]:
            self.best_known_delay = (self.area, self.delay, self.episode, self.iteration)
        if self.delay <= self.params['mapping']['clock_period'] and self.area < self.best_known_area_meets_constraint[0]:
            self.best_known_area_meets_constraint = (self.area, self.delay, self.episode, self.iteration)
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(self.area), str(self.delay)]) + ', ' +
            '; '.join(list(map(str, self.best_known_area_meets_constraint))) + ', ' + 
            '; '.join(list(map(str, self.best_known_area))) + ', ' + 
            '; '.join(list(map(str, self.best_known_delay))) + '\n')
        self.log.flush()

        return new_state, reward, self.iteration == self.params['iterations'], None

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.v')
        output_design_file_mapped = os.path.join(self.episode_dir, str(self.iteration) + '-mapped.v')
    
        abc_command = 'read ' + self.params['mapping']['library_file'] + '; '
        abc_command += 'read ' + self.params['design_file'] + '; '
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'map -D ' + str(self.params['mapping']['clock_period']) + '; '
        abc_command += 'write ' + output_design_file_mapped + '; '
        abc_command += 'topo; stime;'
    
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
        
        ob = re.search(r'Delay *= *[0-9]+.?[0-9]*', line)
        delay = float(ob.group().split('=')[1].strip())
        
        ob = re.search(r'Area *= *[0-9]+.?[0-9]*', line)
        area = float(ob.group().split('=')[1].strip())

        return delay, area
    
    def _get_reward(self, delay, area):
        constraint_met = True
        optimization_improvement = 0    # (-1, 0, 1) <=> (worse, same, improvement)
        constraint_improvement = 0      # (-1, 0, 1) <=> (worse, same, improvement)

        # check optimizing parameter
        if area < self.area:
            optimization_improvement = 1
        elif area == self.area:
            optimization_improvement = 0
        else:
            optimization_improvement = -1
        
        # check constraint parameter
        if delay > self.params["mapping"]["clock_period"]:
            constraint_met = False
            if delay < self.delay:
                constraint_improvement = 1
            elif delay == self.delay:
                constraint_improvement = 0
            else:
                constraint_improvement = -1

        # now calculate the reward
        return self._reward_table(constraint_met, constraint_improvement, optimization_improvement)
    
    def _reward_table(self, constraint_met, contraint_improvement, optimization_improvement):
        return {
            True: {
                0: {
                    1: 3,
                    0: 0,
                    -1: -1
                }
            },
            False: {
                1: {
                    1: 3,
                    0: 2,
                    -1: 1
                },
                0: {
                    1: 2,
                    0: 0,
                    -1: -2
                },
                -1: {
                    1: -1,
                    0: -2,
                    -1: -3
                }
            }
        }[constraint_met][contraint_improvement][optimization_improvement]
    
    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])
    