# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:25:08 2021

@author: murra
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from environment import Dataset
from model import SimpleFly
import matplotlib.pyplot as plt


def OptimizeModel(net, actions, epochs):    
    # Datasets
    trainfunc = Dataset(actions, 10, 1)
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=1, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(net.parameters())
    loss_results = []
    
    # Train
    for epoch in range(epochs):
    
        for k, data in enumerate(trainloader, 0):
            inputs, labels = data[0], data[1]
            net.train()
            optimizer.zero_grad()
            reward = torch.tensor(0.0)
            for i in range(actions):
                X = inputs[:,i,:]
                y = labels[:,i,:]
                action = net(X)
                reward -= torch.squeeze(torch.matmul(y,action.T))
            reward.backward(retain_graph=True)
            optimizer.step()
            loss_results.append(-1*reward.item())
            
            
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    net = SimpleFly()
    results = OptimizeModel(net, 100, 2500)
    plt.plot(results)