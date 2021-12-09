# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:52:50 2021

@author: murra
"""
import torch
import torch.nn as nn
from environment import Dataset
from model import ComplexFly, ComplexFlyLesion
import matplotlib.pyplot as plt
device = torch.device('cpu')

def TestModel(net, batch, actions, threshold):
    net.eval()
    trainfunc = Dataset(actions, threshold, batch)
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=batch, shuffle=False, num_workers=0)
    
    accuracies = []
    for k, data in enumerate(trainloader, 0):
        inputs, contexts, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        net.resetWeights()
        for i in range(actions):
            X = inputs[:,i,:,:]
            c = contexts[:,i,:]
            y = torch.squeeze(labels[:,i,:])
            action = net(X, c)
            y_action = torch.argmax(action, dim=-1)
            acc = torch.sum(torch.eq(y, torch.argmax(y_action, dim=-1))).item()
            accuracies.append(acc*0.01)
        break
            
    return accuracies

if __name__ == "__main__":
    batch = 100
    dt = 0.5
    amount_of_actions = 100
    action_threshold = 5
    net = ComplexFly(batch, dt, device).to(device)
    net.load_state_dict(torch.load('model.pt'))
    results = TestModel(net, batch, amount_of_actions, action_threshold)
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Action number')
    ax.set_title('Accuracy on test set')
    plt.show()
    
    
    
    
    
    