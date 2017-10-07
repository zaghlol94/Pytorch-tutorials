#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:47:33 2017

@author: zaghlol
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
x_values = [i for i in range(20)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim) 

model.load_state_dict(torch.load('linregmodel.pkl'))    
predicted=model(Variable(torch.from_numpy(x_train))).data.numpy
predicted()   
