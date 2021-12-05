# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:25:08 2021

@author: murra
"""
import torch
import torch.nn as nn
from environment import Environment
from model import ComplexFly
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def ObtainData(environment, batch, device):
    Xs = []
    cs = []
    ys = []
    
    for i in range(batch):
        X, c, y = environment.GameOfLife()
        Xs.append(X)
        cs.append(c)
        ys.append(y)
        
    X = torch.stack(Xs, dim=0).to(device)
    c = torch.stack(cs, dim=0).to(device)
    y = torch.stack(ys, dim=0).to(device)
    return X, c, y

def OptimizeModel(net, batch, environment, actions, epochs):    
    # Optimize and Loss
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    lossfunc = nn.CrossEntropyLoss()
    loss_results = []
    
    # Train
    print('Epoch Progress bar')
    for epoch in tqdm(range(epochs)):
        inputs, contexts, labels = ObtainData(environment, batch, device)
        net.resetWeights()
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for i in range(actions):
            X = inputs[:,i,:,:]
            c = contexts[:,i,:]
            y = torch.squeeze(labels[:,i,:])
            action = net(X, c)
            loss += lossfunc(action, y)
        loss.backward()
        optimizer.step()
        loss_results.append(loss.item())
            
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    batch = 30
    dt = 0.5
    amount_of_actions = 100
    action_threshold = 5
    epochs = 200
    net = ComplexFly(batch, dt, device).to(device)
    enviro = Environment(amount_of_actions, action_threshold)
    results = OptimizeModel(net, batch, enviro, amount_of_actions, epochs)
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Cross Entropy Loss')
    ax.set_xlabel('Training sample')
    ax.set_title('Loss for Mushroom Body model')
    plt.show()
    
    
    
    
    
    