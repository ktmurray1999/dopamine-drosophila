# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 06:49:58 2022

@author: murra
"""
import torch
from torch.distributions import Categorical

pnCells = 3

class Environment():
    def __init__(self, theta):
        self.groceries = torch.tensor([[1,0,0],
                                       [1,0,1],
                                       [1,1,1],
                                       [0,1,0],
                                       [0,0,1],
                                       [0,1,1]])
        
        self.classification = torch.tensor([[0],[1],[0],[1],[0],[1],])
        
        self.tables = []
        self.theta = theta
        
    def calc_occupied_probs(self, n):
        probs = []
        for i in self.tables:
            probs.append(i / (n + self.theta))
            
        return probs
        
    def new_customer(self, trial):
        n = trial - 1
        
        if trial == 1:
            self.tables.append(1)
            return 0
        else:
            if len(self.tables) == len(self.classification):
                probs = [i/sum(self.tables) for i in self.tables]
                dist = Categorical(torch.tensor(probs))
                
            else:
                probs = self.calc_occupied_probs(n)
                probs.append(self.theta/(n + self.theta))
                dist = Categorical(torch.tensor(probs))
                
            table = dist.sample()
            if table.item() == len(self.tables):
                self.tables.append(1)
            else:
                self.tables[table.item()] += 1
            
            return table.item()
        
    def get_action(self, trial):
        index = self.new_customer(trial)
        stimuli = self.groceries[index]
        target = self.classification[index]
        
        return stimuli, target
        
    def clear_tables(self,):
        self.tables = []


