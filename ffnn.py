# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:23:43 2019

@author: kka4718
"""

import numpy as np

def sigmoid(x,deriv=False):
    if not deriv:
        return 1/(1+np.e**(-x))
    else:
        return x*(1-x)

class FFNN:
    
    def __init__(self, layout=False,synapsefile=False,neuronfile=False):
        if neuronfile!=False:
            self.neurons=np.loadtxt(neuronfile)
        else:
            self.neurons=np.array([np.zeros(i) for i in layout]) #create neuron layers with zero values and randomised bias
        if synapsefile!=False:      #if savefile exists, load save
            self.synapses=np.loadtxt(synapsefile)
        else:
            self.synapses=np.array([2*np.random.rand(layout[i],layout[i+1])-1 for i in range(len(layout)-1)]) #create synapse layers with randomised weights
    
    def save(self,synapsename,neuronname):  #not yet functional
        np.savetxt(synapsename,self.synapses)
        np.savetxt(neuronname,self.neurons)
    
    def evaluate(self, inputs): #only takes 1d list as inputs
        if len(inputs)!=len(self.neurons[0]): #check for dimensional mismatch
            return 1
        else:
            self.neurons[0]=sigmoid(inputs) #calculate input neuron values
            for e in range(1,len(self.neurons)):
                self.neurons[e]=sigmoid((np.dot(self.neurons[e-1], self.synapses[e-1])).T) #propagate through by using matrix multiplication
        return self.neurons[-1] #return output layer values