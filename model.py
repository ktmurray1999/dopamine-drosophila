# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:23:39 2021

@author: murra
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFly(nn.Module):
    def __init__(self):
        super(SimpleFly, self).__init__()
        self.pn_kc = nn.Linear(50, 2000)
        self.kc_mbon = nn.Linear(2000, 34)
        self.decoder = nn.Linear(34, 2)
        
        self.activation = nn.ReLU()
        self.max = nn.Softmax(dim=-1)

    def forward(self, odor):
        kenyon_cells = self.max(self.pn_kc(odor))
        mbon = self.activation(self.kc_mbon(kenyon_cells))
        action = self.max(self.decoder(mbon))
        return action
    


