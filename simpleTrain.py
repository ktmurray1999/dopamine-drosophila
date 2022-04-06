# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:33:40 2022

@author: murra
"""
import torch
import torch.nn as nn
from simpleEnv import Environment
from simpleModel import simpleFly
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--iter', type=int, default=4)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--actions', type=int, default=31)
parser.add_argument('--theta', type=float, default=0.6)
parser.add_argument('--rate', type=int, default=4)

args = parser.parse_args()

# torch.autograd.set_detect_anomaly(True)

def OptimizeModel(net, batch, environment, epochs, actions,):    
    # Optimize and Loss
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=10**(-1*args.rate))
    lossfunc = nn.CrossEntropyLoss()
    loss_results = []
    
    print('Epoch Progress bar')
    for epoch in tqdm(range(epochs)):
        net.resetWeights()
        life = torch.tensor([[5.0,]])
        loss = torch.tensor(0.0)
        environment.clear_tables()
        optimizer.zero_grad()
        
        for i in range(1,actions):
            stimuli, target = environment.get_action(i, )
            for t in range(15):
                if t < 5:
                    X = torch.tensor([[0,0,0]])
                elif t < 10:
                    X = torch.unsqueeze(stimuli, 0)
                else:
                    X = torch.tensor([[0,0,0]])
                    
                action = net(X, life)
            loss += lossfunc(action, target)
            
            if torch.argmax(action) == target:
                if life.item() != 10.0:
                    life = life + 1.0
            else:
                life = life - 1.0
            
            if life <= 0.0:
                break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        loss_results.append(life.item())
        
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    dt = 0.5
    epochs = args.epochs
    actions = args.actions
    
    net = simpleFly(1, dt,)
    enviro = Environment(args.theta)
    results = OptimizeModel(net, 1, enviro, epochs, actions)
    
    os.mkdir('results'+os.sep+str(args.iter))
    
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Health bar')
    ax.set_title('Final health bar during training epochs')
    plt.savefig('results'+os.sep+str(args.iter)+'/fig_'+str(args.iter)+'.png')
    plt.show()
    
    torch.save(net.state_dict(), 'results'+os.sep+str(args.iter)+'/model.pt')
    torch.save(results, 'results'+os.sep+str(args.iter)+'/results.pt')
    torch.save(args, 'results'+os.sep+str(args.iter)+'/args.pt')
    
    # Testing matrix
    #    Freeze weights and test on all seen stimuli
    # Increase output
    # Does it learn?
    # Make model work
    # Benchmark later
    
