# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:00:37 2019

@author: kka4718
"""

from ffnn import *
import numpy as np
import time


class Backprop_FFNN(FFNN):
    
    def __init__(self, layout=False,savefile=False):
        FFNN.__init__(self,layout,savefile)
    
    def save(self,savename):
        FFNN.save(self,savename)
    
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
    
    
    def train(self,inputs,outputs,alpha=0.3,n=1000,runprint=1000):
        tm=time.time()
        for i in range(n):
            if i%runprint==0:
                print("Training Run No.",i)
            for e in range(len(inputs)):
                self.evaluate(inputs[e])
                a=self.backprop(outputs[e],alpha)
        print(time.time()-tm)