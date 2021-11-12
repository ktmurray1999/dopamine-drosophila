# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:56:52 2021

@author: murra
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ctrnnCell(nn.Module):
    def __init__(self, dim, batch, tau, dt):
        super(ctrnnCell, self).__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.bias = torch.ones(batch, dim)
        self.bias_param = nn.Parameter(torch.tensor(1.0))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state):
        dt_tau = self.dt/self.tau
        state_plus_1 = state*(1-dt_tau) + \
        dt_tau*self.activation(self.linear(state)+self.bias_param*self.bias)
        
        return state_plus_1
    
class ctrnn(nn.Module):
    def __init__(self, dim, batch, time):
        super(ctrnn, self).__init__()
        self.ctrnnCell = ctrnnCell(dim, batch, 10.0, 1.0)
        self.dim = dim
        self.batch = batch
        self.time = time
        
    def forward(self):
        states = torch.zeros(self.time, self.batch, self.dim)
        for i in range(self.time-1):
            states[i+1] = self.ctrnnCell(states[i])
        
        return states
    
def StatePlot(states, dim):
    fig, axs = plt.subplots(dim, 1)
    for i in range(dim):
        axs[i].plot(states[:,0,i])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    net = ctrnn(5, 1, 1000)
    states = net().detach().numpy()
    StatePlot(states, 5)
    
    
    


