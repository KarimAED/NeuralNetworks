# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:36:36 2019

@author: Main
"""

from backprop_ffnn import *
from ScrapeNDict import DB
import numpy as np

class wordEmbedder:
    
    def __init__(self,initparams=["articles1.csv",10,100],savefile=None):
        
        FN,AN,VN=initparams
        self._DB=DB(FN,AN)
        self._Dict=self._DB.getDict()
        self._trainingText=self._DB.getText()
        self._DictDim=len(self._Dict)
        self._VecDim=VN
        if savefile==None:
            self._embedNetwork=Backprop_FFNN(layout=[self._DictDim,self._VecDim,self._DictDim])
        else:
            self._embedNetwork=Backprop_FFNN(savefile=savefile)
    
    def genTrainingData(self,n):
        self._trainingData=[]
        for i in range(n):
            for j in range(len(self._trainingText)):
                tempR=0
                while tempR==0 or j+tempR+1>len(self._trainingText) or j+tempR<0:
                    tempR=np.random.randint(-3,3)
                self._trainingData.append([self._Dict.index((self._trainingText[j]).lower()),self._Dict.index((self._trainingText[j+tempR]).lower())])
        self._trainingData=np.array(self._trainingData).T
        return self._trainingData
    
    def train(self,datan=1,alpha=0.3,runn=1000):
        self.genTrainingData(datan)
        ips=[]
        ops=[]
        for i in range(len(self._trainingData[0])):
            temp1=np.zeros(self._DictDim)
            temp2=np.zeros(self._DictDim)
            temp1[self._trainingData[0][i]]=1
            temp2[self._trainingData[1][i]]=1
            ips.append(temp1)
            ops.append(temp2)
        ips=np.array(ips)
        ops=np.array(ops)
        self._embedNetwork.train(ips,ops,alpha,runn,1)
    
    def relate(self,inputWord):
        N=self._Dict.index(inputWord)
        ip=np.zeros(self._DictDim)
        ip[N]=1
        op=self._embedNetwork.evaluate(ip)
        return self._Dict(np.argmax(op))
    
    def save(self,savename):
        self._embedNetwork.save(savename)