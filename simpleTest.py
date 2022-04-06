# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:30:42 2022

@author: murra
"""
import torch
import numpy as np
# from simpleEnv import Environment
from simpleModel import simpleFly
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

theta = 0.75
test_folder = '0'
test_iters = 3

groceries = torch.tensor([[1,0,0], [1,0,1], [1,1,1], 
                          [0,1,0], [0,0,1], [0,1,1]])
classification = torch.tensor([[0],[1],[0],[1],[0],[1],])
index = np.arange(len(classification)) + 0.3
bar_width = 0.4

def TestModel(model, test_iters):
    life = torch.tensor([[5.0,]])
    results = [[], [], [], [], [], [],]
    # model.train()
    # environment.clear_tables()
    
    for j in tqdm(range(test_iters)):
        for i in range(len(classification)):
            Stimuli = groceries[i:i+1]
            Class = classification[i:i+1]
            model.resetWeights()
            
            for t in range(15):
                if t < 5:
                    X = torch.tensor([[0,0,0]])
                elif t < 10:
                    X = Stimuli
                else:
                    X = torch.tensor([[0,0,0]])
                
                action = model(X, life)
            print(action)
            action = torch.argmax(torch.squeeze(action))
            
            if action == Class:
                results[i].append(1)
            else:
                results[i].append(0)
    
    bar_graphs = []
    for i in results:
        ave = sum(i)/len(i)
        bar_graphs.append(ave)
    
    return bar_graphs


if __name__ == "__main__":
    dt = 0.5
    
    model = simpleFly(1, dt,)
    model.load_state_dict(torch.load('results/'+test_folder+'/model.pt'))
    
    bar_heights = TestModel(model, test_iters)
    
    os.mkdir('results/test_'+test_folder)
    
    fig, ax = plt.subplots()
    ax.bar(index, bar_heights, bar_width)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Health bar')
    ax.set_title('Final health bar during training epochs')
    plt.savefig('results/test_'+test_folder+'/testing_results.png')
    plt.show()
    
    
    
    