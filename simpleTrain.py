# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:33:40 2022

@author: murra
"""
import torch
from simpleEnv import Environment
from model import simpleFly
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device('cpu')

# torch.autograd.set_detect_anomaly(True)

def OptimizeModel(net, batch, environment, epochs, actions, time):    
    # Optimize and Loss
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    loss_results = []
    
    print('Epoch Progress bar')
    for epoch in tqdm(range(epochs)):
        net.resetWeights()
        life = torch.tensor([[5.0,]]) # Saturate health bar
        environment.clear_tables()
        optimizer.zero_grad()
        
        for i in range(actions):
            stimuli, target = environment.get_action(i, )
            for t in range(time):
                if t < 5:
                    X = torch.tensor([[0,0,0]])
                elif t < 10:
                    X = torch.unsqueeze(stimuli, 0)
                else:
                    X = torch.tensor([[0,0,0]])
                    
                action = net(X, torch.tensor([[life.item(),]]))
            life += torch.sum(torch.squeeze(action)*target) # Use cross entropy loss function
            
            if life < 0.0: # This is only a stopping condition, not a learning condition
                break
        
        loss = -1*life
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        loss_results.append(life.item())
        
    print('Finished Training')
    return loss_results


if __name__ == "__main__":
    batch = 1
    dt = 0.5
    epochs = 50
    actions = 10 # Increase actions
    time = 15
    
    net = simpleFly(batch, dt, device)
    enviro = Environment()
    results = OptimizeModel(net, batch, enviro, epochs, actions, time)
    
    fig, ax = plt.subplots()
    ax.plot(results)
    ax.set_ylabel('Training Health bar')
    ax.set_xlabel('Training epoch')
    ax.set_title('Health bar of model during training epochs')
    plt.savefig('results.png')
    plt.show()
    torch.save(net.state_dict(), 'model_simple.pt')
    
    # Testing matrix
    #    Freeze weights and test on all seen stimuli
    # Increase output
    # Does it learn?
    # Make model work
    # Benchmark later
    
