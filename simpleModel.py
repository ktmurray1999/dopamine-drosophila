# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:20:18 2022

@author: murra
"""
import torch
import torch.nn as nn

pnCells = 3
knCells = 50
mbonCells = 10
danCells = 10
    
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
        pn_value = torch.matmul(pn_inputs.type(torch.FloatTensor), self.pn_weight)
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
        self.kc_weight = nn.Parameter(torch.ones(knCells, 1))
        self.bias = nn.Parameter(torch.tensor([-0.5]))
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, kc_inputs):
        dt_tau = self.dt/self.tau
        kc_value = torch.matmul(kc_inputs, self.kc_weight)
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(kc_value + self.bias)
        
        return state_plus_1

class MBONCell(nn.Module):
    def __init__(self, tau, tau_weights, batch, dt):
        super(MBONCell, self).__init__()
        self.dan_weight = torch.distributions.normal.Normal(torch.zeros(danCells*mbonCells), 
                                                           torch.tensor([0.5])).sample()
        self.dan_weight = nn.Parameter(torch.reshape(self.dan_weight, (danCells, mbonCells)))
        
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(mbonCells), 
                                                                   torch.tensor([0.5])).sample())
        self.tau_weights = torch.tensor(tau_weights)
        self.tau = torch.tensor(tau)
        self.batch = batch
        self.dt = torch.tensor(dt)
        
        self.dan_activation = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, state, kc_weights, kc_inputs, dan_inputs):
        dt_tau = self.dt/self.tau
        kc_value = torch.matmul(torch.unsqueeze(kc_inputs, 1), kc_weights)
        kc_value = torch.squeeze(kc_value)
        dan_value = torch.matmul(dan_inputs, self.dan_weight)
        dan_mod = self.dan_activation(torch.squeeze(dan_value))
        state_plus_1 = state*(1-dt_tau) + dt_tau*dan_mod*self.activation(kc_value + self.bias)
        
        return state_plus_1
    
    def changeWeights(self, weights, weight_activations, kc, kc_low, dan, dan_low):
        dt_tau = self.dt/self.tau_weights # Think about adjusting this
        synaptic_activity = torch.unsqueeze(kc,-1)*torch.unsqueeze(dan_low,1) - torch.unsqueeze(kc_low,-1)*torch.unsqueeze(dan,1)
        weight_activations_plus_1 = weight_activations + self.dt*(synaptic_activity)
        weights_plus_1 = weights*(1-dt_tau) + dt_tau*weight_activations
        
        return weights_plus_1, weight_activations_plus_1

class DANCell(nn.Module):
    def __init__(self, tau, RC, dt):
        super(DANCell, self).__init__()
        self.mbon_weight = torch.distributions.normal.Normal(torch.zeros(mbonCells*danCells), 
                                                             torch.tensor([0.5])).sample()
        self.mbon_weight = nn.Parameter(torch.reshape(self.mbon_weight, (mbonCells, danCells)))
        
        self.dan_weight = torch.distributions.normal.Normal(torch.zeros(danCells*danCells), 
                                                             torch.tensor([0.5])).sample()
        self.dan_weight = nn.Parameter(torch.reshape(self.dan_weight, (danCells, danCells)))
        
        self.context_weight = torch.distributions.normal.Normal(torch.zeros(1*danCells),
                                                                torch.tensor([0.5])).sample()
        self.context_weight = nn.Parameter(torch.reshape(self.context_weight, (1, danCells)))
        self.bias = nn.Parameter(torch.distributions.normal.Normal(torch.zeros(danCells), 
                                                                   torch.tensor([0.5])).sample())
        self.RC = torch.tensor(RC)
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        
        self.activation = nn.ReLU()

    def forward(self, state, mbon_inputs, context):
        dt_tau = self.dt/self.tau
        mbon_value = torch.matmul(mbon_inputs, self.mbon_weight)
        context_value = torch.matmul(context, self.context_weight)
        dan_state_recurrent = torch.matmul(state, self.dan_weight)
        state_plus_1 = state*(1-dt_tau) + dt_tau*self.activation(dan_state_recurrent + 
                                                                 mbon_value + 
                                                                 context_value + 
                                                                 self.bias)
        
        return state_plus_1
    
    def lowPass(self, state, LowPassState):
        alpha = self.dt/(self.dt + self.RC)
        LowPassState_plus_1 = alpha*state + (1-alpha)*LowPassState
        
        return LowPassState_plus_1

class simpleFly(nn.Module):
    def __init__(self, batch, dt,):
        super(simpleFly, self).__init__()
        self.kenyon = KenyonCell(1,5,batch,dt)
        self.apl = APLCell(1,dt)
        self.mbon = MBONCell(1,2,batch,dt)
        self.dan = DANCell(1,5,dt)
        self.decoder = nn.Linear(mbonCells, 2)
        
        self.state_pn = torch.zeros(batch, pnCells)
        self.state_kc = torch.zeros(batch, knCells)
        self.state_apl = torch.zeros(batch, 1)
        self.state_mbon = torch.zeros(batch, mbonCells)
        self.state_dan = torch.zeros(batch, danCells)
        
        self.state_low_kc = torch.zeros(batch, knCells)
        self.state_low_dan = torch.zeros(batch, danCells)
        self.state_weight_activation = torch.zeros(batch, knCells, mbonCells)
        
        self.kc_weight = torch.distributions.normal.Normal(torch.zeros(batch*knCells*mbonCells), 
                                                           torch.tensor([0.5])).sample()
        self.kc_weight = torch.reshape(self.kc_weight, (batch, knCells, mbonCells))
        
        self.batch = batch
        self.dt = dt
        
    def forward(self, odor, context):        
        self.state_kc = self.kenyon(self.state_kc, odor, self.state_apl)
        self.state_apl = self.apl(self.state_apl, self.state_kc)
        self.state_mbon = self.mbon(self.state_mbon, self.kc_weight, self.state_kc, self.state_dan)
        self.state_dan = self.dan(self.state_dan, self.state_mbon, context)
        
        self.state_low_kc = self.kenyon.lowPass(self.state_kc, self.state_low_kc)
        self.state_low_dan = self.dan.lowPass(self.state_dan, self.state_low_dan)
        
        self.kc_weight, self.state_weight_activation = self.mbon.changeWeights(self.kc_weight, 
                                                                          self.state_weight_activation, 
                                                                          self.state_kc, 
                                                                          self.state_low_kc,
                                                                          self.state_dan, 
                                                                          self.state_low_dan)
            
        output = self.decoder(self.state_mbon)
        
        return output
    
    def resetWeights(self,):
        self.state_pn = torch.zeros(self.batch, pnCells)
        self.state_kc = torch.zeros(self.batch, knCells)
        self.state_apl = torch.zeros(self.batch, 1)
        self.state_mbon = torch.zeros(self.batch, mbonCells)
        self.state_dan = torch.zeros(self.batch, danCells)
        
        self.state_low_kc = torch.zeros(self.batch, knCells)
        self.state_low_dan = torch.zeros(self.batch, danCells)
        self.state_weight_activation = torch.zeros(self.batch, knCells, mbonCells)

        self.kc_weight = torch.distributions.normal.Normal(torch.zeros(self.batch*knCells*mbonCells), 
                                                           torch.tensor([0.5])).sample()
        self.kc_weight = torch.reshape(self.kc_weight, (self.batch, knCells, mbonCells))
        # self.kenyon.resetWeights()
        
    def forward_no_reset(self, odor, context):
        self.state_kc = self.kenyon(self.state_kc, odor, self.state_apl)
        self.state_apl = self.apl(self.state_apl, self.state_kc)
        self.state_mbon = self.mbon(self.state_mbon, self.kc_weight, self.state_kc, self.state_dan)
        self.state_dan = self.dan(self.state_dan, self.state_mbon, context)
            
        output = self.decoder(self.state_mbon)
        
        return output
    
    def reset_not_forget(self,):
        self.state_pn = torch.zeros(self.batch, pnCells)
        self.state_kc = torch.zeros(self.batch, knCells)
        self.state_apl = torch.zeros(self.batch, 1)
        self.state_mbon = torch.zeros(self.batch, mbonCells)
        self.state_dan = torch.zeros(self.batch, danCells)    
    
    