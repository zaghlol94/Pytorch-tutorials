#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:59:23 2017

@author: zaghlol
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

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
criterion = nn.MSELoss()   
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 130

for epoch in range(epochs):
    epoch += 1
    inputs=Variable(torch.from_numpy(x_train))
    labels=Variable(torch.from_numpy(y_train))
    optimizer.zero_grad()  
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))
    
predicted=model(Variable(torch.from_numpy(x_train))).data.numpy
predicted()   

torch.save(model.state_dict(),'linregmodel.pkl')