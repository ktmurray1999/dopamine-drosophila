# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:29:51 2021

@author: murra
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

context = {'hungry': 0, 'thirsty': 1, 'tired':2}
context_lookup = {0: 'hungry', 1: 'thirsty', 2:'tired'}
actions = {'consume': 0, 'rest': 1}
actions_lookup = {0: 'consume', 1: 'rest'}

context_change_p = [1/3, 1/3, 1/3]

hungry_reward = [[1.0,0.0],
                 [0.0,1.0]]

thirsty_reward = [[0.0,1.0],
                  [1.0,0.0]]

tired_reward = [[0.0,1.0],
                [0.0,1.0]]

reward_signal = torch.tensor([hungry_reward,thirsty_reward,tired_reward])

context_signal = torch.tensor([[1,0,0],
                               [0,1,0],
                               [0,0,1]])

class Environment():
    def __init__(self, total_actions, action_threshold):
        self.threshold = action_threshold
        self.context = np.random.choice(len(context), 1, p=context_change_p)[0]
        self.total_actions = total_actions
        
        self.actions = 0
        self.context_switches = 0
        
        self.context_odor_action_reward = reward_signal
        self.context_signals = context_signal
        
        self.upper_dist = torch.distributions.bernoulli.Bernoulli(0.9*torch.ones(24))
        self.lower_dist = torch.distributions.bernoulli.Bernoulli(0.1*torch.ones(23))
            
    def environment_update(self):
        self.actions += 1
        if self.actions > self.threshold:
            self.context = np.random.choice(len(context), 1, p=context_change_p)[0]
            self.actions = 0
            self.context_switches += 1
        
    def decision(self):
        odor = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        
        if odor == 0:
            odor_tensor = torch.cat((self.upper_dist.sample(),self.lower_dist.sample()), 0)
        elif odor == 1:
            odor_tensor = torch.cat((self.lower_dist.sample(),self.upper_dist.sample()), 0)
        odor_tensor = torch.cat((odor_tensor, self.context_signals[self.context]), 0)
        
        reward = self.context_odor_action_reward[self.context, odor]
        
        self.environment_update()
        
        return odor_tensor, reward
    
    def GameOfLife(self):
        X = torch.zeros(self.total_actions, 50)
        y = torch.zeros(self.total_actions, 2)
        
        for i in range(self.total_actions):
            odor_tensor, reward = self.decision()
            
            X[i,:] = odor_tensor
            y[i,:] = reward
        
        return X, y
        
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, actions, threshold, runs):
        self.actions = actions
        self.threshold = threshold
        self.runs = runs
        
    def __len__(self):
        return self.runs
    
    def __getitem__(self, index):
        enviro = Environment(self.actions, self.threshold)
        X, y = enviro.GameOfLife()
        return X, y
    
    
    
    



