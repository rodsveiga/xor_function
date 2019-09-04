#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:20:55 2019

@author: rodsveiga
"""

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
np.random.seed(12)



class NeuralNetwork:
    #########
    # parameters
    # ----------
    # self:      the class object itself
    # net_arch:  consists of a list of integers, indicating
    #            the number of neurons in each layer, i.e. the network architecture
    #########
    def __init__(self, 
                 net_arch,
                 act= 'sig',
                 beta= 1.0,
                 W= None, 
                 epsilon= 0):
        
                
        def sig(x, beta):
            return 1.0 / (1.0 + np.exp(-beta*x))

        def sig_derivative(x, beta):
            return beta*x*(1.0 - x)
        
        def tanh(x, beta):
            return (1.0 - np.exp(-2*beta*x))/(1.0 + np.exp(-2*beta*x))

        def tanh_derivative(x, beta):
            return beta*(1.0 + x)*(1.0 - x)  
        
      
        
        if act == 'sig':
            self.activity = sig
            self.activity_derivative = sig_derivative
            
        if act == 'tanh':
            self.activity = tanh
            self.activity_derivative = tanh_derivative
            
        # Initialized the weights, making sure we also 
        # initialize the weights for the biases that we will add later
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.beta = beta
        self.int_rep_ = []
        self.saved_weights = []
        
        if epsilon == 0:

            if W is None:
                self.weights = []
                # Random initialization with range of weight values (-1,1)
                for layer in range(self.layers - 1):
                    w_ = 2*np.random.rand(net_arch[layer] + 1, net_arch[layer+1]) - 1
                    self.weights.append(w_)
            else:
                self.weights = W
                
        else:
                
            ## Perturbation over the cross weights
            for layer in range(self.layers -1):
                w_ = np.identity(net_arch[layer])
                w_ = w_ + epsilon
                bias = np.zeros([1, net_arch[layer]])
                w_ = np.concatenate([w_, bias])       
                self.weights.append(w_)
                    
    
    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weights)-1):
            activation = np.dot(y[i], self.weights[i])
            activity = self.activity(activation, self.beta)

            # add the bias for the next layer ### CHECK THOSE BIAS
            activity = np.concatenate((np.ones(1), np.array(activity)))
            y.append(activity)

        # last layer
        activation = np.dot(y[-1], self.weights[-1])
        activity = self.activity(activation, self.beta)
        y.append(activity)
        
        return y
    
    
    def forward(self, x):
        
        ones = np.ones((1, x.shape[0]))
        Z = np.concatenate((ones.T, x), axis=1)

              
        int_ = []
        int_.append(x)
        
        for i in range(len(self.weights)-1):
            
            activation = np.dot(Z, self.weights[i])
            activity = self.activity(activation, self.beta)
            
            int_.append(activity)
            
            Z = activity
            Z = np.concatenate((ones.T, Z), axis=1)
            
        # Last layer
        activation = np.dot(Z, self.weights[-1])
        activity = self.activity(activation, self.beta)
        int_.append(activity)
        
        return int_
        
    def _back_prop(self, y, target, learning_rate):
        
        error = target - y[-1]
        delta_vec = [error * self.activity_derivative(y[-1], self.beta)]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error*self.activity_derivative(y[i][1:], self.beta)
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()
        
        # Finally, we adjust the weights, using the backpropagation rules
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i]+1)
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            self.weights[i] += learning_rate*layer.T.dot(delta)
            
    
    #########
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, 
            data,
            labels,
            learning_rate= 0.1,
            epochs= 100,
            int_rep= False,
            save_weights= False,
            int_rep_index= -2):
        
        # Add bias units to the input layer - 
        # add a "1" to the input data (the always-on bias neuron)
        ones = np.ones((1, data.shape[0]))
        Z = np.concatenate((ones.T, data), axis=1)
        
        # Last layer
        self.int_rep_.append(data)
        # All layers
        #self.int_rep_.append(self.forward(data))
        
        self.saved_weights.append(self.weights)
        
        for k in range(epochs):
            if (k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))
        
            sample = np.random.randint(data.shape[0])

            # We will now go ahead and set up our feed-forward propagation:
            x = [Z[sample]]
            y = self._forward_prop(x)

            # Now we do our back-propagation of the error to adjust the weights:
            target = labels[sample]
            self._back_prop(y, target, learning_rate)

            # Internal monitor
            if int_rep:
                
                # Last layer
                #lin_bias = np.dot(Z, self.weights[int_rep_index])
                #act = self.activity(lin_bias, self.beta)
                #self.int_rep_.append(act)
                # All layers 
                int_ = self.forward(data)
                self.int_rep_.append(int_)
          
            if save_weights:
                self.saved_weights.append(self.weights)
                      
                
                
            
    def int_rep_dyn(self):
        return self.int_rep_
    
    def weights_dyn(self):
        return self.saved_weights
          
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    
    def predict_single_data(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(np.dot(val, self.weights[i]), self.beta)
            val = np.concatenate((np.ones(1).T, np.array(val)))
        return val[1]
    
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # X:      the input data array
    #########
    def predict(self, X):
        Y = np.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = np.array([[self.predict_single_data(x)]])
            Y = np.vstack((Y,y))
        return Y
    
    
    
    
    ############################################
    
    
    
    
class Plots():
    
    def __init__(self, 
                 X,
                 y,
                 classifier):
        
        self.X = X
        self.y = y
        self.classifier = classifier
                   
    def decision_regions(self,
                         test_idx=None, 
                         resolution=0.02):
                
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])
    
        # plot the decision surface
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    
        # plot class samples
        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x= self.X[self.y == cl, 0], y= self.X[self.y == cl, 1],
                        alpha= 0.8, c= cmap(idx),
                        marker=markers[idx], label= cl)
    
        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = self.X[test_idx, :], self.y[test_idx]
    
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c= y_test,
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')