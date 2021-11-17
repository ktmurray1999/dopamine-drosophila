# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:25:08 2021

@author: murra
"""
import torch
import torch.nn as nn
from environment import Dataset
from model import ComplexFly
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def OptimizeModel(net, batch, actions, threshold, epochs):    
    # Datasets
    trainfunc = Dataset(actions, threshold, batch*10)
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=batch, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(net.parameters())
    lossfunc = nn.CrossEntropyLoss()
    loss_results = []
    
    # Train
    for epoch in range(epochs):
    
        for k, data in enumerate(trainloader, 0):
            inputs, contexts, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            net.train()
            net.resetWeights()
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for i in range(actions):
                X = inputs[:,i,:,:]
                c = contexts[:,i,:]
                y = torch.squeeze(labels[:,i,:])
                action = net(X, c)
                loss += lossfunc(action, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_results.append(loss.item())
        print(loss.item())
            
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    batch = 30
    dt = 0.5
    amount_of_actions = 100
    action_threshold = 5
    epochs = 500
    net = ComplexFly(batch, dt, device).to(device)
    results = OptimizeModel(net, batch, amount_of_actions, action_threshold, epochs)
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Cross Entropy Loss')
    ax.set_xlabel('Training sample')
    ax.set_title('Loss for Mushroom Body model')
    plt.show()
    
    
    
    
    
    