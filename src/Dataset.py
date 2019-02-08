#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.util import random_noise
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
        self.corruptionLevel = data["corruptionLevel"]
        self.dataLength = len(self.train_depth)
        if data["colorSpace"] == "BGR":
            self.colorSpace = None
            self.colorSpaceRevert = None
            
        if data["colorSpace"] == "RGB":
            self.colorSpace = cv2.COLOR_BGR2RGB
            self.colorSpaceRevert = cv2.COLOR_RGB2BGR
        elif data["colorSpace"]== "HSV":
            self.colorSpace = cv2.COLOR_BGR2HSV
            self.colorSpaceRevert = cv2.COLOR_HSV2BGR
        elif data["colorSpace"]== "LUV":
            self.colorSpace = cv2.COLOR_BGR2LUV
            self.colorSpaceRevert = cv2.COLOR_LUV2BGR
        elif data["colorSpace"]== "LAB":
            self.colorSpace = cv2.COLOR_BGR2LAB
            self.colorSpaceRevert = cv2.COLOR_LAB2BGR
        elif data["colorSpace"]== "LAB":
            self.colorSpace = cv2.COLOR_BGR2LAB
            self.colorSpaceRevert = cv2.COLOR_LAB2BGR
        elif data["colorSpace"]== "YCrCb":
            self.colorSpace = cv2.COLOR_BGR2YCrCb
            self.colorSpaceRevert = cv2.COLOR_YCrCb2BGR
        elif data["colorSpace"]== "HLS":
            self.colorSpace = cv2.COLOR_BGR2HLS
            self.colorSpaceRevert = cv2.COLOR_HLS2BGR
        elif data["colorSpace"]== "XYZ":
            self.colorSpace = cv2.COLOR_BGR2XYZ
            self.colorSpaceRevert = cv2.COLOR_XYZ2BGR
        elif data["colorSpace"]== "YUV":
            self.colorSpace = cv2.COLOR_BGR2YUV
            self.colorSpaceRevert = cv2.COLOR_YUV2BGR
            
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
                
    def loadImages(self,imgs,colorConversion, corruption):
        img_list=[]
        for i in imgs:
            img=cv2.imread(i)
            if corruption == True :
                img = random_noise(img/255.0,mode='s&p',amount= self.corruptionLevel)
                img = np.uint8(img*255.)     
            if colorConversion:
                img=cv2.cvtColor(img,self.colorSpace)
            img=cv2.resize(img,(self.imgWidth,self.imgHeight))
            if not img is None:
                img_list.append(img)
            else: 
                continue
        img_list = self.preProcessImages(img_list)
        return img_list
    
    def postProcessImages(self,images):
        if self.colorSpaceRevert is None :
            return images
        list_images = [cv2.cvtColor(np.uint8(img),self.colorSpaceRevert) for img in images]
        return list_images
    
    def preProcessImages(self,img_list):
        #img_list=[np.float32(i/255.0) for i in img_list]
        img_list=np.asarray(img_list)
        img_list=np.float32(img_list.reshape(-1,self.imgHeight,self.imgWidth,self.imgChannels))
        return img_list
        
    def nextTrainBatch(self,corruptionFlag):
        inp=self.train_depth[self.start:self.end]
        gt=self.train_rgb[self.start:self.end]
        inp=self.loadImages(inp,False, False)
        gt=self.loadImages(gt,True,corruptionFlag)
        self.start=self.end
        self.end+=self.batchSize
        if self.end >= len(self.train_depth):
            self.epoch+=1
            print("************** Training data : EPOCH {} COMPLETED************\n\n".format(self.epoch))
            self.start=0
            self.end=self.batchSize
            self.train_depth,self.train_rgb=shuffle(self.train_depth,self.train_rgb)
        return (inp,gt)    
    def resetTrainBatch(self):
        self.epoch = 0
        self.start = 0
        self.end   = self.batchSize
        print("Train batch handlers reset")
        
    def resetTestBatch(self):
        self.test_epoch=0
        self.test_start=0
        self.test_end=self.batchSize
        print ("Test batch handlers reset")
        
    def nextTestBatch(self,corruptionFlag):
        inp=self.test_depth[self.test_start:self.test_end]
        gt=self.test_rgb[self.test_start:self.test_end]
        inp=self.loadImages(inp,False,False)
        gt=self.loadImages(gt,True,corruptionFlag)
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


# In[16]:


if __name__ == '__main__':
    data = {"scale":1,"batchSize":4,"train_file" : "/home/kgunase3/data/NYUD/RAW/train.txt",
            "test_file" : '/home/kgunase3/data/NYUD/RAW/train.txt', "colorSpace":"RGB", 
            "imageWidth" : 640,"imageHeight" :480, "channels":3,"corruptionLevel" : 0.3}
    dataObj = dataReader(data)
    inp,gt = dataObj.nextTrainBatch(corruptionFlag= True)
    print (gt.shape)


# In[17]:


import matplotlib.pyplot as plt
plt.imshow(np.uint8(gt[0]))
plt.figure()
plt.imshow(np.uint8(gt[1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




