# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:00:37 2019

@author: kka4718
"""

from ffnn import *
import numpy as np

class Backprop_FFNN(FFNN):
    
    def __init__(self, layout=False,synapsefile=False,neuronfile=False):
        FFNN.__init__(self,layout,synapsefile,neuronfile)
    
    def save(self,synapsename,neuronname):
        FFNN.save(self,synapsename,neuronname)
    
    def evaluate(self, inputs):
        return FFNN.evaluate(self,inputs)
    
    
    def backprop(self,outputs,alpha):  #backpropagation of the error to adjust weights
        if len(outputs)!=len(self.neurons[-1]): #check for dimensional mismatch
            return 1
        else:
            self.err=np.array([outputs-self.neurons[-1]]) #find output error
            initerr=self.err
            for i in range(1,len(self.neurons)):  #loop through the network towards the input layer
                self.delta=alpha*self.err*sigmoid(self.neurons[-i],deriv=True) #adjust for certainty of our prediction and use alpha coefficient to prevent overshoot
                self.synapses[-i]+=np.dot(np.array([self.neurons[-(i+1)]]).T,self.delta) #calculate the error due to the weights in the current synapse layer using gradients
                self.err=np.dot(self.delta,self.synapses[-i].T) #calculate the error coming from the previous layer to propagate through again and again until all layers are readjusted
            return initerr #return error to give useful data to user
    
    def train(self,inputs,outputs,alpha=0.1,n=100):
        for i in range(n):
            if i%100==0:
                print(i)
            for e in range(len(inputs)):
                self.evaluate(inputs[e])
                a=self.backprop(outputs[e],alpha)
                if i%10000==0:
                    print(a)
                    pass