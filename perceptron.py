#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:08:08 2019

@author: rodsveiga
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

class Perceptron():
    
    def __init__(self,
                 X,
                 y):
        
        self.X = X
        self.y = y
        
        self.boundary_lines = []
                 
        self.x_min= min(X.T[0])
        self.x_max= max(X.T[0])
        self.y_min= max(X.T[0])
        self.y_max= max(X.T[0])
        
    
    
    def stepFunction(self, 
                     x):
        
        return np.heaviside(x, 1).astype(int)
    
    
    def prediction(self, 
                   X, 
                   W, 
                   b):
        return self.stepFunction((np.matmul(X,W)+b))
    
    # The function should receive as inputs the data X, the labels y,
    # the weights W (as an array), and the bias b,
    # update the weights and bias W, b, according to the perceptron algorithm,
    # and return W and b.
    def perceptronStep(self, 
                       X, 
                       y,
                       W, 
                       b, 
                       learn_rate):
        for i in range(len(X)):
            y_hat = self.prediction(X[i], W, b)[0]
            if y[i]-y_hat == 1:
                W[0] += X[i][0]*learn_rate
                W[1] += X[i][1]*learn_rate
                b += learn_rate
            elif y[i]-y_hat == -1:
                W[0] -= X[i][0]*learn_rate
                W[1] -= X[i][1]*learn_rate
                b -= learn_rate
        return W, b
    
    # This function runs the perceptron algorithm repeatedly on the dataset,
    # and returns a few of the boundary lines obtained in the iterations,
    # for plotting purposes.
    # Feel free to play with the learning rate and the num_epochs,
    # and see your results plotted below.
    def train(self,
              learn_rate = 0.01, 
              num_epochs = 10000):

        W = np.array(np.random.rand(len(self.X[0]),1))
        b = np.random.rand(1)[0] + self.x_max
        
        for i in range(num_epochs):
            # In each epoch, we apply the perceptron step.
            W, b = self.perceptronStep(self.X, self.y, W, b, learn_rate)
            self.boundary_lines.append((-W[0]/W[1], -b/W[1]))
        
            y_pred = self.prediction(self.X, W, b)
            
            lin_sep = np.array_equal(y_pred.reshape(len(self.y)), self.y)
            
            if lin_sep:
                break
        
        if i == num_epochs - 1:
            output = False
            
        output = True
        
        return output
    
    def plot(self):
        
        gap = 0.25
        
        x_domain = np.linspace(self.x_min - gap, self.x_max + gap)

        lines = np.array(self.boundary_lines)

        for j in range(len(lines)):
            plt.plot(x_domain, x_domain*lines[j, 0] + lines[j, 1], color= 'r', linestyle=':') 
            plt.plot(x_domain, x_domain*lines[len(lines)-1, 0] + lines[len(lines)-1, 1], linestyle='-', color='g') 
            plt.scatter(self.X[:, 0], self.X[:, 1], c = self.y)

        plt.ylim(self.x_min - gap, self.x_max + gap)
        plt.title('Solution Boundary')

        plt.show()