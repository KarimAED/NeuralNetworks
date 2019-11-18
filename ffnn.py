# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:23:43 2019

@author: kka4718
"""

import numpy as np
import os
import shutil

def sigmoid(x,deriv=False):
    if not deriv:
        return 1/(1+np.e**(-x))
    else:
        return x*(1-x)


class FFNN:
    
    def __init__(self, layout=False,savefile=False):
        if savefile!=False: #check for existing save
            path=os.getcwd()
            reg=os.scandir(path+"\\"+savefile)
            os.chdir(path+"\\"+savefile)
            self.neurons=[]
            self.synapses=[]
            for i in reg:
                if "_neurons" in i.name:
                    self.neurons.append(np.loadtxt(i.name))
                elif "_synapses" in i.name:
                    self.synapses.append(np.loadtxt(i.name))
            self.neurons=np.array(self.neurons)
            self.synapses=np.array(self.synapses)
        else:
            self.neurons=np.array([np.zeros(i) for i in layout]) #create neuron layers with zero values and randomised bias
            self.synapses=np.array([2*np.random.rand(layout[i],layout[i+1])-1 for i in range(len(layout)-1)]) #create synapse layers with randomised weights
           
    
    def save(self,savename):  #not yet functional
        path=os.getcwd()
        try:
            os.mkdir(path+"\\"+savename)
        except OSError:
            if input("Do you want to override the existing network?(Y/N) ")=="Y":
                os.chdir(path)
                shutil.rmtree(path+"\\"+savename)
                os.mkdir(path+"\\"+savename)
            else:
                print("Save Aborted!")
        os.chdir(path+"\\"+savename)
        print(os.getcwd())
        for i in range(len(self.neurons)):
            np.savetxt(savename+"_neurons"+str(i)+".txt.gz",self.neurons[i],header="Neurons of Layer "+str(i))
        for i in range(len(self.synapses)):
            np.savetxt(savename+"_synapses"+str(i)+".txt.gz",self.synapses[i],header="Neurons of Layer "+str(i))
        print("Save successful!")
        os.chdir(path)
            
    
    
    def evaluate(self, inputs): #only takes 1d list as inputs
        if len(inputs)!=len(self.neurons[0]): #check for dimensional mismatch
            return 1
        else:
            self.neurons[0]=sigmoid(inputs) #calculate input neuron values
            for e in range(1,len(self.neurons)):
                self.neurons[e]=sigmoid((np.dot(self.neurons[e-1], self.synapses[e-1])).T) #propagate through by using matrix multiplication
        return self.neurons[-1] #return output layer values