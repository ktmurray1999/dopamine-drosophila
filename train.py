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
device = torch.device('cpu')

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

def FindNan(net):
    value = False
    state_dict = net.state_dict()
    for i in state_dict:
        if torch.sum(torch.isnan(state_dict[i])).item() > 0:
            string_to_exec = 'net.'+i+'.data = torch.nan_to_num(state_dict[i], nan=-0.5)'
            exec(string_to_exec)
            print(i)
            print(torch.sum(torch.isnan(state_dict[i])).item())
            print('---------------------')
            value = True
    
    return value

def OptimizeModel(net, batch, environment, actions, epochs):    
    # Optimize and Loss
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    lossfunc = nn.CrossEntropyLoss()
    loss_results = []
    
    switch = True
    hold = net.state_dict()
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
        value = FindNan(net)
        if value and switch:
            torch.save(hold, 'hold_model.pt')
            torch.save(net.state_dict(), 'bad_model.pt')
            switch = False
        hold = net.state_dict()
        
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    batch = 10
    dt = 0.5
    amount_of_actions = 100
    action_threshold = 20
    epochs = 2500
    net = ComplexFly(batch, dt, device).to(device)
    net.load_state_dict(torch.load('model_complex.pt'))
    enviro = Environment(amount_of_actions, action_threshold)
    enviro_foods = torch.load('environment_vectors.pt')
    enviro.odor_food = enviro_foods['odor_food']
    enviro.odor_drink = enviro_foods['odor_drink']
    results = OptimizeModel(net, batch, enviro, amount_of_actions, epochs)
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Cross Entropy Loss')
    ax.set_xlabel('Training sample')
    ax.set_title('Loss for Mushroom Body model')
    plt.savefig('results.png')
    plt.show()
    torch.save({'results':results}, 'results_12_9_p2.pt')
    torch.save(net.state_dict(), 'model_complex1.pt')
    
    # enviro_vecs = {'odor_food': enviro.odor_food, 'odor_drink': enviro.odor_drink}
    # torch.save(enviro_vecs, 'environment_vectors.pt')
    
    
    