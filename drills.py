#!/usr/bin/python3
import yaml
import os
import subprocess
import sys
import re
import tensorflow as tf
import numpy as np

data_file = 'data.yml'
with open(data_file, 'r') as f:
    options = yaml.load(f)

def log(message):
    if not os.path.exists(options['output_dir']):
        os.makedirs(options['output_dir'])
    print(message)
    with open(os.path.join(options['output_dir'], 'agent.log'), 'a') as f:
        f.write(message + '\n')

class Game:
    def __init__(self, options):
        self.options = options
        self.design_file = options['design_file']
        self.current_design_file = self.design_file
        self.library_file = options['mapping']['library_file']
        self.clock_period = options['mapping']['clock_period']
        self.abc_binary = options['abc_binary']

        self.optimizations = options['optimizations']
        self.iterations = optiosn['iterations']

        self.action_space_length = len(self.optimizations)
        self.observation_space_size = 4     # the output of print_stats command

        self.iteration = 0
        self.num_of_episodes = 0
        self.delay = float('inf')
        self.minimum_delay = float('inf')
    
    def reset(self, episode):
        """
        resets the environment and returns the state
        """ 
        self.current_design_file = self.design_file
        self.iteration = 0
        self.delay = float('inf')
        self.episode_dir = os.path.join(options['output_dir'], str(episode))
        if not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)

        return self.run_optimization('map -D 150')
    
    def step(self, action):
        """
        accepts action (which is optimization index) and returns (new state, reward, done, info)
        """
        previous_delay = self.delay
        optimization = self.optimizations[action]

        log_message = 'Iteration: ' + str(self.iteration) + '\n'
        log_message += 'Current Delay: ' + str(self.delay) + '\n'
        log_message += 'Optimization: ' + str(optimization) + '\n'

        new_state = self.run_optimization(optimization)
        reward = - (self.delay - previous_delay)
        self.iteration += 1

        log_message += 'New Delay: ' + str(self.delay) + '\n'
        log_message += 'Reward: ' + str(reward) + '\n'
        log_message += 'State of system: ' + ','.join(map(str, new_state)) + '\n'
        
        log(log_message)
        return new_state, reward, self.iteration == min(self.num_of_episodes + 1, self.iterations), None
    
    def get_state(self, stats):
        """
        extracts observation space from print_stats result
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        
        ob = re.search(r'delay *= *[1-9]+.?[0-9]+', line)
        delay = float(ob.group().split('=')[1].strip())
        
        ob = re.search(r'area *= *[1-9]+.?[0-9]+', line)
        area = float(ob.group().split('=')[1].strip())
        
        ob = re.search(r'nd *= *[1-9]+.?[0-9]+', line)
        nd = float(ob.group().split('=')[1].strip())

        ob = re.search(r'edge *= *[1-9]+.?[0-9]+', line)
        edge = float(ob.group().split('=')[1].strip())

        ob = re.search(r'lev *= *[1-9]+.?[0-9]+', line)
        lev = float(ob.group().split('=')[1].strip())
        
        # state representation needs to be on the same scale for the NN to train well
        # EDIT: write a function to do autoscaling of the parameters
        return np.array([delay/100, area/100000, nd/100000, edge/100000])
    
    def run_optimization(self, optimization):
        """
        returns new_design_file, state
        """
        output_dir = os.path.join(self.episode_dir, str(self.iteration), optimization.replace(' ', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_design_file = os.path.join(output_dir, 'design.blif')
    
        abc_command = 'read ' + self.library_file + '; '
        abc_command += 'read ' + self.current_design_file + '; '
        abc_command += 'strash; '
        abc_command += optimization + '; '
        abc_command += 'map -D ' + str(self.clock_period) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'print_stats; '
    
        try:
            proc = subprocess.check_output([self.abc_binary, '-c', abc_command])
            self.state = self.get_state(proc)
            self.current_design_file = output_design_file    
            self.delay = self.state[0] * 100.0
            self.minimum_delay = min([self.delay, self.minimum_delay])
        except Exception as e:
            print(e)
            pass
        return self.state    

class A2C:
    def __init__(self, options):
        self.game = Game(options)

        self.num_actions = self.game.action_space_length
        self.state_size = self.game.observation_space_size

        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])

        # Define any additional placeholders needed for training your agent here:
        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ])

        self.state_value = self.critic()
        self.actor_probs = self.actor()
        self.loss_val = self.loss()
        self.train_op = self.optimizer()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
        self.gamma = 0.99
        self.learning_rate = 0.01

    def optimizer(self):
        """
        :return: Optimizer for your loss function
        """
        return tf.train.AdamOptimizer(0.01).minimize(self.loss_val)        

    def critic(self):
        """
        Calculates the estimated value for every state in self.state_input. The critic should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states] representing the estimated value of each state in the trajectory.
        """
        c_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    
        c_fc2 = tf.contrib.layers.fully_connected(inputs=c_fc1,
                                                num_outputs=1,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
        
        return c_fc2

    def actor(self):
        """
        Calculates the action probabilities for every state in self.state_input. The actor should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states, num_actions] representing the probability distribution
            over actions that is generated by your actor.
        """
        a_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
        a_fc2 = tf.contrib.layers.fully_connected(inputs=a_fc1,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
        
        a_fc3 = tf.contrib.layers.fully_connected(inputs=a_fc2,
                                                num_outputs=self.num_actions,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
        return tf.nn.softmax(a_fc3)

    def loss(self):
        """
        :return: A scalar tensor representing the combined actor and critic loss.
        """
        # critic loss
        advantage = self.discounted_episode_rewards_ - self.state_value
        critic_loss = tf.reduce_sum(tf.square(advantage))

        # actor loss        
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.log(self.actor_probs), 
                                                                  labels=self.actions)
        actor_loss = tf.reduce_sum(neg_log_prob * advantage)
        
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actor_probs,
                                                                 labels=self.actions)
        policy_gradient_loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_)
        # return policy_gradient_loss
        
        return critic_loss + actor_loss

    def train_episode(self, episode_number):
        """
        train_episode will be called 1000 times by the autograder to train your agent. In this method,
        run your agent for a single episode, then use that data to train your agent. Feel free to
        add any return values to this method.
        """
        state = self.game.reset(episode_number)
        self.game.num_of_episodes += 1
        done = False
        log('Episode: ' + str(episode_number))
        log('--------------')
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not done:
            action_probability_distribution = self.session.run(self.actor_probs,
                                                               feed_dict={self.state_input: state.reshape([1, self.state_size])})
            print(action_probability_distribution)
            action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                      p=action_probability_distribution.ravel())
            new_state, reward, done, _ = self.game.step(action)
            
            # append this step
            episode_states.append(state)
            action_ = np.zeros(self.num_actions)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)
            
            state = new_state
        
        # Now that we have run the episode, we use this data to train the agent
        discounted_episode_rewards = self.discount_and_normalize_rewards(episode_rewards)
        
        _ = self.session.run(self.train_op, feed_dict={self.state_input: np.array(episode_states),
                                                        self.actions: np.array(episode_actions),
                                                        self.discounted_episode_rewards_: discounted_episode_rewards})
        
        log('==================')
        log('Episode ended with total rewards: ' + str(np.sum(episode_rewards)))
        log('Episode minimum delay was: ' + str(self.game.minimum_delay))
        return np.sum(episode_rewards)
    
    
    def discount_and_normalize_rewards(self, episode_rewards):
        """
        used internally to calculate the discounted episode rewards
        """
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
    
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
    
        discounted_episode_rewards = (discounted_episode_rewards - mean) / std
    
        return discounted_episode_rewards

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
    log('Mean reward over the last 100 episodes = ' + str(mean_reward))

