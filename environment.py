# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:29:51 2021

@author: murra
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

context = {'light': 0, 'dark': 1}
context_lookup = {0: 'light', 1: 'dark'}
actions = {'eat': 0, 'rest': 1}
actions_lookup = {0: 'eat', 1: 'rest'}
states = {'hungry': 0, 'satiated': 1, 'tired': 2}
states_lookup = {0: 'hungry', 1: 'satiated', 2: 'tired'}

light_eat_states =  [[0.00, 1.00, 0.00],
                     [0.00, 0.20, 0.80],
                     [0.00, 0.00, 1.00]]
light_eat_rewards = [[0.00, 1.00, 0.00],
                     [0.00, 1.00, 0.00],
                     [0.00, 1.00, 0.00]]
light_rest_states = [[1.00, 0.00, 0.00],
                     [0.20, 0.80, 0.00],
                     [0.70, 0.20, 0.10]]
light_rest_rewards =[[-1.00, 0.00, 0.00],
                     [0.00, 0.00, 0.00],
                     [0.00, 0.00, 0.00]]

dark_eat_states =  [[0.00, 1.00, 0.00],
                    [0.00, 0.20, 0.80],
                    [0.00, 0.00, 1.00]]
dark_eat_rewards = [[-1.00, 1.00, -1.00],
                    [-1.00, 1.00, -1.00],
                    [-1.00, 1.00, -1.00]]
dark_rest_states = [[1.00, 0.00, 0.00],
                    [0.20, 0.80, 0.00],
                    [0.70, 0.20, 0.10]]
dark_rest_rewards =[[0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00]]

all_states = np.array([[light_eat_states, light_rest_states],
                       [dark_eat_states, dark_rest_states]])
all_rewards = np.array([[light_eat_rewards, light_rest_rewards],
                        [dark_eat_rewards, dark_rest_rewards]])

class Environment():
    def __init__(self, init_state, action_threshold):
        self.threshold = action_threshold
        self.state = init_state
        self.context = 0
        self.day = 0
        
        self.actions = 0
        self.reward = 0
        
        self.state_transitions = all_states
        self.reward_transitions = all_rewards
            
    def environment_update(self):
        if self.actions > self.threshold and self.context == 0:
            self.context = 1
            self.actions = 0
        elif self.actions > self.threshold and self.context == 1:
            self.context = 0
            self.actions = 0
            self.day += 1
        
    def decision(self, act):
        probabilities = self.state_transitions[self.context, act, self.state]
        new_state = np.random.choice(len(states), 1, p=probabilities)[0]
        given_reward = self.reward_transitions[self.context, act, self.state, new_state]
        self.state = new_state
        self.reward += given_reward
        
        self.actions += 1
        self.environment_update()
        
        return given_reward
        
    
if __name__ == "__main__":
    enviro = Environment(2, 5)
    
    while enviro.day < 5:
        print('You are '+states_lookup[enviro.state])
        print('It is '+context_lookup[enviro.context])
        act = input('Do you eat or rest?  ')
        reward = enviro.decision(actions[act])
        print('You got a reward of '+str(reward))
        print('\n')

    print('End with reward of '+str(enviro.reward))



