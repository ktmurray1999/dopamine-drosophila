# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:23:39 2021

@author: murra
"""
import torch
import torch.nn as nn

pnCells = 20
knCells = 800
outCells = 14

class ProjectionNeuron(nn.Module):
    def __init__(self, tau, dt):
        super(ProjectionNeuron, self).__init__()
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(pnCells), 
                                                                   torch.tensor([0.5])).sample())
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, inputs):
        dt_tau = self.dt/self.tau
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(inputs + self.bias)
        
        return state_plus_1
    
class KenyonCell(nn.Module):
    def __init__(self, tau, RC, batch, dt):
        super(KenyonCell, self).__init__()
        self.pn_weight = torch.distributions.normal.Normal(torch.zeros(batch*pnCells*knCells), 
                                                           torch.tensor([0.5])).sample()
        self.pn_weight = torch.reshape(self.pn_weight, (batch, pnCells, knCells))
        self.apl_weight = torch.ones(1, knCells)
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(knCells), 
                                                                   torch.tensor([0.5])).sample())
        self.RC = torch.tensor(RC)
        self.tau = torch.tensor(tau)
        self.batch = batch
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, pn_inputs, apl_inputs):
        dt_tau = self.dt/self.tau
        apl_value = torch.matmul(apl_inputs, self.apl_weight)
        pn_value = torch.matmul(torch.unsqueeze(pn_inputs, 1), self.pn_weight)
        pn_value = torch.squeeze(pn_value)
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(pn_value - apl_value + self.bias)
        
        return state_plus_1
    
    def lowPass(self, state, LowPassState):
        alpha = self.dt/(self.dt + self.RC)
        LowPassState_plus_1 = alpha*state + (1-alpha)*LowPassState
        
        return LowPassState_plus_1
    
    def resetWeights(self):
        self.pn_weight = torch.distributions.normal.Normal(torch.zeros(self.batch*pnCells*knCells), 
                                                           torch.tensor([0.5])).sample()
        self.pn_weight = torch.reshape(self.pn_weight, (self.batch, pnCells, knCells))

class APLCell(nn.Module):
    def __init__(self, tau, dt):
        super(APLCell, self).__init__()
        self.kc_weight = torch.ones(knCells, 1)
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(1), 
                                                                   torch.tensor([0.5])).sample())
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, kc_inputs):
        dt_tau = self.dt/self.tau
        kc_value = torch.matmul(kc_inputs, self.kc_weight)
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(kc_value + self.bias)
        
        return state_plus_1

class MBONCell(nn.Module):
    def __init__(self, tau, tau_weights, dt):
        super(MBONCell, self).__init__()
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(outCells), 
                                                                   torch.tensor([0.5])).sample())
        self.tau_weights = torch.tensor(tau_weights)
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.dan_activation = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, state, kc_weights, kc_inputs, dan_inputs):
        dt_tau = self.dt/self.tau
        kc_value = torch.matmul(torch.unsqueeze(kc_inputs, 1), kc_weights)
        kc_value = torch.squeeze(kc_value)
        dan_mod = self.dan_activation(dan_inputs)
        state_plus_1 = state*(1-dt_tau) + dt_tau*dan_mod*self.activation(kc_value + self.bias)
        
        return state_plus_1
    
    def changeWeights(self, weights, weight_activations, kc, kc_low, dan, dan_low):
        dt_tau = self.dt/self.tau_weights
        synaptic_activity = torch.unsqueeze(kc,-1)*torch.unsqueeze(dan_low,1) - torch.unsqueeze(kc_low,-1)*torch.unsqueeze(dan,1)
        weight_activations_plus_1 = weight_activations + self.dt*(synaptic_activity)
        weights_plus_1 = weights*(1-dt_tau) + dt_tau*weight_activations
        
        return weights_plus_1, weight_activations_plus_1

class DANCell(nn.Module):
    def __init__(self, tau, RC, dt):
        super(DANCell, self).__init__()
        self.mbon_weight = torch.distributions.normal.Normal(torch.zeros(outCells*outCells), 
                                                             torch.tensor([0.5])).sample()
        self.mbon_weight = nn.Parameter(torch.reshape(self.mbon_weight, (outCells, outCells)))
        self.context_weight = torch.distributions.normal.Normal(torch.zeros(3*outCells),
                                                                torch.tensor([0.5])).sample()
        self.context_weight = nn.Parameter(torch.reshape(self.context_weight, (3, outCells)))
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(outCells), 
                                                                   torch.tensor([0.5])).sample())
        self.RC = torch.tensor(RC)
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, mbon_inputs, context):
        dt_tau = self.dt/self.tau
        mbon_value = torch.matmul(mbon_inputs, self.mbon_weight)
        context_value = torch.matmul(context, self.context_weight)
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(mbon_value + context_value + self.bias)
        
        return state_plus_1
    
    def lowPass(self, state, LowPassState):
        alpha = self.dt/(self.dt + self.RC)
        LowPassState_plus_1 = alpha*state + (1-alpha)*LowPassState
        
        return LowPassState_plus_1

class ComplexFly(nn.Module):
    def __init__(self, batch, dt, device):
        super(ComplexFly, self).__init__()
        self.projection = ProjectionNeuron(1,dt).to(device)
        self.kenyon = KenyonCell(1,5,batch,dt).to(device)
        self.apl = APLCell(2.5,dt).to(device)
        self.mbon = MBONCell(1,5,dt).to(device)
        self.dan = DANCell(1,5,dt).to(device)
        self.decoder = nn.Linear(outCells, 2).to(device)
        
        self.kc_weight = torch.distributions.normal.Normal(torch.zeros(batch*knCells*outCells), 
                                                           torch.tensor([0.5])).sample()
        self.kc_weight = torch.reshape(self.kc_weight, (batch, knCells, outCells)).to(device)
                
        self.batch = batch
        self.dt = dt
        self.device = device

    def forward(self, odor, context):
        state_pn = torch.zeros(self.batch, pnCells).to(self.device)
        state_kc = torch.zeros(self.batch, knCells).to(self.device)
        state_apl = torch.zeros(self.batch, 1).to(self.device)
        state_mbon = torch.zeros(self.batch, outCells).to(self.device)
        state_dan = torch.zeros(self.batch, outCells).to(self.device)
        
        state_low_kc = torch.zeros(self.batch, knCells).to(self.device)
        state_low_dan = torch.zeros(self.batch, outCells).to(self.device)
        state_weight_activation = torch.zeros(self.batch, knCells, outCells).to(self.device)
        
        for t in range(odor.size(1)):
            state_pn = self.projection(state_pn, odor[:,t])
            state_kc = self.kenyon(state_kc, state_pn, state_apl)
            state_apl = self.apl(state_apl, state_kc)
            state_mbon = self.mbon(state_mbon, self.kc_weight, state_kc, state_dan)
            state_dan = self.dan(state_dan, state_mbon, context)
            
            state_low_kc = self.kenyon.lowPass(state_kc, state_low_kc)
            state_low_dan = self.dan.lowPass(state_dan, state_low_dan)
            
            self.kc_weight, state_weight_activation = self.mbon.changeWeights(self.kc_weight, 
                                                                              state_weight_activation, 
                                                                              state_kc, state_low_kc,
                                                                              state_dan, state_low_dan)
            
        output = self.decoder(state_mbon)
        
        return output
    
    def resetWeights(self,):
        self.kc_weight = torch.distributions.normal.Normal(torch.zeros(self.batch*knCells*outCells), 
                                                           torch.tensor([0.5])).sample()
        self.kc_weight = torch.reshape(self.kc_weight, (self.batch, knCells, outCells)).to(self.device)
        self.kenyon.resetWeights()




