# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:01:01 2019

@author: kka4718
"""

class DB:
    
    def __init__(self,fname,samplesize):
        
        self._DBName=fname
        self.loadDB(samplesize)

    def loadDB(self,lastline):
        self.loadFile(lastline)
        self.genText()
        self.genDict()
        print("\nWord Count of Sources:",len(self._Text),"\nDictionary Size:",len(self._Dict))
    
    def loadFile(self,lastline):
        DB_raw=open(self._DBName,"r",encoding="utf8")
        print("Opened File.\nReading Data...")
        fl=[]
        for i in range(lastline+1):
            fl.append(DB_raw.readline())
        DB_raw.close()
        print("Closed File.")
        articles_raw=[i.split("\"") for i in fl if i!=fl[0]]
        self._Articles=[]
        for j in articles_raw:
            if len(j)>3:
                self._Articles.append(j[3])
            else:
                try:
                    self._Articles.append(j[1])
                except:
                    pass
        print("Finished Gathering Raw Data.")
    
    def genText(self):
        print("Started Processing Text...")
        self._exclude=",.;:“”-—|()0123456789£$€!?&’‘•…■¡♥»¼½"
        Dlong=""
        for d in self._Articles:
            Dlong+=d
        Data=Dlong.split()
        self._Text=[e.strip(self._exclude) for e in Data if e.strip(self._exclude)!=""]
        print("Finished Processing Text.")
    
    def genDict(self):
        print("Generating Dictionary...")
        self._Dict=[]
        for i in self._Text:
            if not i.lower() in self._Dict:
                self._Dict.append(i.lower())
        self._Dict=sorted(self._Dict)
        print("Finished Dictionary.")
    
    def findDict(self,string):
        self._Results=[]
        for i in range(len(self._Dict)):
            if string in self._Dict[i]:
                self._Results.append([i,self._Dict[i]])
        return self._Results
    
    def getName(self):
        return self._DBName
    
    def getRaw(self):
        return self._Articles
    
    def getText(self):
        return self._Text
    
    def getDict(self):
        return self._Dict
    
    def getLastSearch(self):
        try:
            return self._Results
        except:
            print("No previous search conducted!")
            return 0
