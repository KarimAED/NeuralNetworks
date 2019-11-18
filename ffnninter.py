# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:34:06 2019

@author: Karim
"""

from ffnn import *
import numpy as np
from tkinter import *


def colorconverter(x):
    if x>1:
        x=1
    elif x<-1:
        x=-1
    rgb=[255,255,255]
    if x<0:
        y=int(abs(x)*255)
        rgb[1]-=y
        rgb[2]-=y
    else:
        y=int(x*255)
        rgb[0]-=y
        rgb[2]-=y
    rgb=tuple(rgb)
    return "#%02x%02x%02x" %rgb

class NNinter(FFNN):

    def __init__(self,layout=False,savefile=False):
        super(NNinter,self).__init__(layout,savefile)
        self.master=Tk()
        self.w,self.h=(640,480)
        self.scr=Canvas(master=self.master,bg="white",width=self.w,height=self.h)
        self.scr.pack()
        self.xstep=self.w/(len(self.neurons)+1)
        self.draw()
    
    def draw(self):
        
        self.scr.delete(ALL)
        
        for e in range(len(self.neurons)):
            x=self.xstep*(e+1)
            ystep=self.h/(len(self.neurons[e])+1)
            for i in range(len(self.neurons[e])):
                y=ystep*(i+1)
                self.scr.create_oval(x-5,y-5,x+5,y+5,fill=colorconverter(self.neurons[e][i]))
        
        for i in range(len(self.synapses)):
            x1=self.xstep*(i+1)
            x2=x1+self.xstep
            ystep1=self.h/(len(self.neurons[i])+1)
            ystep2=self.h/(len(self.neurons[i+1])+1)
            for j in range(len(self.synapses[i])):
                for k in range(len(self.synapses[i][j])):
                    y1=ystep1*(j+1)
                    y2=ystep2*(k+1)
                    self.scr.create_line(x1,y1,x2,y2,fill=colorconverter(self.synapses[i][j][k]))
        
        self.scr.update()

    def save(self,savename):
        super(NNinter,self).save(savename)
    
    def evaluate(self,inputs,draw=False):
        temp=super(NNinter,self).evaluate(inputs)
        if draw!=False:
            self.draw()
        return temp
    
