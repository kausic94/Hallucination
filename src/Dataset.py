#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#%matplotlib inline


# In[2]:


class dataReader():
    def __init__(self,data):
        print ("Initializing Data Reader")
        self.scale=data["scale"]
        self.batchSize=data["batchSize"]
        self.epoch=0
        self.start=0
        self.end=self.batchSize
        self.test_epoch=0
        self.test_start=0
        self.test_end=self.batchSize
        self.train_depth,self.train_rgb=[],[]
        self.test_depth,self.test_rgb =[],[]
        self.loadTrain(data["train_file"])
        self.loadTest(data["test_file"])
        self.imgWidth=data["imageWidth"]
        self.imgHeight=data["imageHeight"]
        self.imgChannels=data["channels"]
        self.dataLength = len(self.train_depth)
        assert len(self.train_depth)==len(self.train_rgb),"Inconsistent length of training input and output"
        assert len(self.test_depth) == len(self.test_rgb),"Inconsistent length of testing input and output"
        print ("Train files {}".format(len(self.train_depth)))
        print ("Test  files {}".format(len(self.test_depth)))
        print ("Initialization Complete")

    def loadTrain(self,train_f):
        with open(train_f,'r') as fh:
            for i in fh:
                data=i.split(',')
                self.train_depth.append(data[0])
                self.train_rgb.append(data[1][:-1]) # exclude the final \n

    def loadTest(self,test_f):
        with open(test_f,'r') as fh :
            for i in fh:
                data=i.split(',')
                self.test_depth.append(data[0])
                self.test_rgb.append(data[1][:-1]) # exclude the final \n
                
    def loadImages(self,imgs):
        img_list=[]
        for i in imgs:
            img=cv2.imread(i)
            img=cv2.resize(img,(self.imgWidth,self.imgHeight))
            if not img is None:
                img_list.append(img)
            else: 
                continue
        img_list = self.preProcessImages(img_list)
        return img_list
    
    def preProcessImages(self,img_list):
        #img_list=[np.float32(i/255.0) for i in img_list]
        img_list=np.asarray(img_list)
        img_list=np.float32(img_list.reshape(-1,self.imgHeight,self.imgWidth,self.imgChannels))
        return img_list
        
    def nextTrainBatch(self):
        inp=self.train_depth[self.start:self.end]
        gt=self.train_rgb[self.start:self.end]
        inp=self.loadImages(inp)
        gt=self.loadImages(gt)
        self.start=self.end
        self.end+=self.batchSize
        if self.end >= len(self.train_depth):
            self.epoch+=1
            print("************** Training data : EPOCH {} COMPLETED************\n\n".format(self.epoch))
            self.start=0
            self.end=self.batchSize
            self.train_depth,self.train_rgb=shuffle(self.train_depth,self.train_rgb)
        return (inp,gt)

    def nextTestBatch(self):
        inp=self.test_depth[self.test_start:self.test_end]
        gt=self.test_rgb[self.test_start:self.test_end]
        inp=self.loadImages(inp)
        gt=self.loadImages(gt)
        self.test_start=self.test_end
        self.test_end+=self.batchSize
        if self.test_end >= len(self.test_depth):
            self.test_epoch+=1
            print("*************Testing data : EPOCH {} COMPLETED ************ \n\n".format(self.test_epoch))
            self.test_start=0
            self.test_end=self.batchSize
            self.test_depth,self.test_rgb = shuffle(self.test_depth,self.test_rgb)
        return (inp,gt)
    
    def vizRandom(self): #Randomly visualize data in training and testing datasets
        ind=np.random.randint(0,len(self.test_depth))
        train_in=cv2.imread(self.train_depth[ind])
        train_in=cv2.cvtColor(train_in,cv2.COLOR_BGR2RGB)
        train_out=cv2.imread(self.train_rgb[ind])
        train_out=cv2.cvtColor(train_out,cv2.COLOR_BGR2RGB)
        test_in=cv2.imread(self.test_depth[ind])
        test_in=cv2.cvtColor(test_in,cv2.COLOR_BGR2RGB)
        test_out=cv2.imread(self.test_rgb[ind])
        test_out=cv2.cvtColor(test_out,cv2.COLOR_BGR2RGB)
        f,ax=plt.subplots(2,2)
        ax[0][0].set_title("Train Input")
        ax[0][0].imshow(train_in)
        ax[0][1].set_title("Train Output")
        ax[0][1].imshow(train_out)
        ax[1][0].set_title("Test Input")
        ax[1][0].imshow(test_in)
        ax[1][1].set_title("Test Output")
        ax[1][1].imshow(test_out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




