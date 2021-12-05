# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:29:51 2021

@author: murra
"""
import numpy as np
import torch

context = {'hungry': 0, 'thirsty': 1, 'tired':2}
context_lookup = {0: 'hungry', 1: 'thirsty', 2:'tired'}
actions = {'consume': 0, 'rest': 1}
actions_lookup = {0: 'consume', 1: 'rest'}

context_change_p = [1/3, 1/3, 1/3]

hungry_reward = [0,1]
thirsty_reward = [1,0]
tired_reward = [1,1]
reward_signal = torch.tensor([hungry_reward,
                              thirsty_reward,
                              tired_reward])

context_signal = torch.tensor([[1,0,0],
                               [0,1,0],
                               [0,0,1]])

pnCells = 20

class Environment():
    def __init__(self, total_actions, action_threshold):
        self.threshold = action_threshold
        self.context = np.random.choice(len(context), 1, p=context_change_p)[0]
        self.total_actions = total_actions
        
        self.actions = 0
        self.context_switches = 0
        
        self.context_odor_action_reward = reward_signal
        self.context_signals = context_signal
        self.create_odors()
        
    def create_odors(self,):
        self.dist = torch.distributions.bernoulli.Bernoulli(0.5*torch.ones(pnCells))
        
        self.odor_food = [self.dist.sample() for i in range(4)]
        self.odor_drink = [self.dist.sample() for i in range(4)]
        
    def environment_update(self):
        self.actions += 1
        if self.actions > self.threshold:
            self.context = np.random.choice(len(context), 1, p=context_change_p)[0]
            self.actions = 0
            self.context_switches += 1
        
    def decision(self):
        odor = np.random.randint(0, 2, 1)[0]
        odor_index =  np.random.randint(0, 4, 1)[0]
        
        if odor == 0:
            odor_tensor = self.odor_food[odor_index]
        elif odor == 1:
            odor_tensor = self.odor_drink[odor_index]
        
        reward = self.context_odor_action_reward[self.context, odor]
        
        self.environment_update()
        
        return odor_tensor, reward, self.context_signals[self.context]
    
    def GameOfLife(self):
        X = torch.zeros(self.total_actions, 25, pnCells)
        c = torch.zeros(self.total_actions, 3)
        y = torch.zeros(self.total_actions, 1, dtype=torch.long)
        
        for i in range(self.total_actions):
            odor_tensor, reward, contexts = self.decision()
            
            X[i,10:15,:] = odor_tensor*torch.ones(5,pnCells)
            c[i,:] = contexts
            y[i,:] = reward
        
        return X, c, y






