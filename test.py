# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:52:50 2021

@author: murra
"""
import torch
import torch.nn as nn
from environment import Environment
from train import ObtainData
from model import ComplexFly
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device('cpu')

def TestModel(net, batch, environment, actions):    
    # Optimize and Loss
    net.eval()
    accuracies = []
    
    # Train
    inputs, contexts, labels = ObtainData(environment, batch, device)
    net.resetWeights()
    for i in range(actions):
        X = inputs[:,i,:,:]
        c = contexts[:,i,:]
        y = torch.squeeze(labels[:,i,:])
        action = net(X, c)
        y_action = torch.argmax(action, dim=-1)
        acc = torch.sum(torch.eq(y, torch.argmax(y_action, dim=-1))).item()
        accuracies.append(acc*0.01)
            
    return accuracies

if __name__ == "__main__":
    batch = 100
    dt = 0.5
    amount_of_actions = 200
    action_threshold = 20
    net = ComplexFly(batch, dt, device).to(device)
    net.load_state_dict(torch.load('model_complex1.pt'))
    enviro = Environment(amount_of_actions, action_threshold)
    enviro_foods = torch.load('environment_vectors.pt')
    enviro.odor_food = enviro_foods['odor_food']
    enviro.odor_drink = enviro_foods['odor_drink']
    results = TestModel(net, batch, enviro, amount_of_actions)
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Action number')
    ax.set_title('Accuracy on test set')
    plt.savefig('accuracy.png')
    plt.show()
    
    
    
    
    
    