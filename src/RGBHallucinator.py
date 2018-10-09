
# coding: utf-8

# In[7]:

import numpy as np
import cv2
import tensorflow as tf
import Dataset
import time
import os
import configparser as ConfigParser
import sys
import argparse 
import psutil


# In[10]:

class Hallucinator ():    
    def __init__ (self,config_file,scale,gpu_num):
        print ("Initializing Hallucinator class")
        self.scale=scale
        self.gpu ="/gpu:{}".format(gpu_num)
        self.readConfiguration(config_file)
        self.depth=tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='depthInput')             
        self.rgb  =tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='grountTruth')
        self.dataObj = Dataset.dataReader(self.dataArguments)
        if not os.path.exists(self.summary_writer_dir):
            os.makedirs(self.summary_writer_dir)
        if not os.path.exists(self.modelLocation):
            os.makedirs(self.modelLocation)
        logPath = os.path.join(self.logDir,self.modelName)
        if not os.path.exists(logPath):
            os.makedirs(logPath)
        self.logDir = os.path.join(logPath,'log.txt')
        self.logger = open(self.logDir,'w')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        self.sess = None
        
    def readConfiguration(self,config_file):
        print ("Reading configuration File")
        config = ConfigParser.ConfigParser()
        config.read(config_file)
        self.imageWidth=int(int(config.get('DATA','imageWidth'))/self.scale)
        self.imageHeight=int(int(config.get('DATA','imageHeight'))/self.scale)
        self.channels=int (config.get('DATA','channels'))
        self.train_file = config.get ('DATA','train_file')
        self.test_file  = config.get ('DATA','test_file')
        self.batchSize  = int(config.get ('DATA','batchSize'))
        self.dataArguments = {"imageWidth":self.imageWidth,"imageHeight":self.imageHeight,                              "channels" : self.channels, "batchSize":self.batchSize,                              "train_file":self.train_file,"test_file":self.test_file,"scale":self.scale}    
        self.maxEpoch=int(config.get('TRAIN','maxEpoch'))
        self.learningRate = float(config.get('TRAIN','learningRate'))
        
        self.print_freq=int(config.get('LOG','print_freq'))
        self.save_freq = int (config.get('LOG','save_freq'))
        self.val_freq = int (config.get('LOG','val_freq'))
        self.modelName = config.get('LOG','modelName') +"_s{}".format(self.scale)
        self.modelLocation = config.get('LOG','modelLocation')
        self.modelLocation = os.path.join(self.modelLocation , self.modelName)
        self.checkPoint =  bool(int(config.get('LOG','checkpoint')))
        self.restoreModelPath =config.get('LOG','restoreModelPath')
        self.logDir = config.get('LOG','logFile')

        if self.checkPoint:
            print ("Using the latest trained model in check point file")
            self.restoreModelPath = tf.train.latest_checkpoint(self.restoreModelPath)
        self.summary_writer_dir =os.path.join(config.get('LOG','summary_writer_dir') ,self.modelName)     
        
        
    def generateImage(self):  # Inference procedure
        with tf.variable_scope(self.modelName, reuse=tf.AUTO_REUSE):
            #layer 1
            conv1 = tf.layers.conv2d(inputs=self.depth,filters=147,kernel_size=(11,11), padding="same",name="conv1",kernel_initializer=tf.truncated_normal_initializer)
            conv1 = tf.contrib.layers.instance_norm(inputs=conv1)
            conv1 = tf.nn.selu(conv1,name="conv1_actvn")
            
            #layer 2
            conv2 = tf.layers.conv2d(inputs=conv1,filters=36,kernel_size=(11,11),padding="same",name="conv2",kernel_initializer=tf.truncated_normal_initializer)
            conv2 = tf.contrib.layers.instance_norm(inputs=conv2)
            conv2 = tf.nn.selu(conv2,name='conv2_actvn')
            
            #layer 3
            conv3 = tf.layers.conv2d(inputs=conv2,filters=36,kernel_size=(11,11),padding="same",name="conv3",kernel_initializer=tf.truncated_normal_initializer)
            conv3 = tf.contrib.layers.instance_norm(inputs=conv3)
            conv3 = tf.nn.selu(conv3,name='conv3_actvn')
            
            #layer 4
            conv4 =  tf.layers.conv2d(inputs=conv3,filters=36,kernel_size=(11,11),padding="same",name="conv4",kernel_initializer=tf.truncated_normal_initializer)
            conv4 = tf.contrib.layers.instance_norm(inputs=conv4)
            conv4 = tf.nn.selu(conv4,name="conv4_actvn")
            
            #layer 5
            conv5 = tf.layers.conv2d(inputs=conv4,filters=36,kernel_size=(11,11),padding="same",name="conv5",kernel_initializer=tf.truncated_normal_initializer)
            conv5 = tf.contrib.layers.instance_norm(inputs=conv5)
            conv5 = tf.nn.selu(conv5,name="conv5_actvn")
            
            #layer 6 
            conv6 = tf.layers.conv2d(inputs=conv5,filters=36,kernel_size=(11,11),padding="same",name="conv6",kernel_initializer=tf.truncated_normal_initializer)
            conv6 = tf.contrib.layers.instance_norm(inputs=conv6)
            conv6 = tf.nn.selu(conv6,name="conv6_actvn")
            
            #layer 7
            conv7 = tf.layers.conv2d(inputs=conv6,filters=36,kernel_size=(11,11),padding="same",name="conv7",kernel_initializer=tf.truncated_normal_initializer)
            conv7 = tf.contrib.layers.instance_norm(inputs=conv7)
            conv7 = tf.nn.selu(conv7,name="conv7_actvn")
            
            #layer 8
            conv8 = tf.layers.conv2d(inputs=conv7,filters=36,kernel_size=(11,11),padding="same",name="conv8",kernel_initializer=tf.truncated_normal_initializer)
            conv8 = tf.contrib.layers.instance_norm(inputs=conv8)
            conv8 = tf.nn.selu(conv8,name="conv8_actvn")
            
            #layer 9 
            conv9 = tf.layers.conv2d(inputs=conv8,filters=36,kernel_size=(11,11),padding="same",name="conv9",kernel_initializer=tf.truncated_normal_initializer)
            conv9 = tf.contrib.layers.instance_norm(inputs=conv9)
            conv9 = tf.nn.selu(conv9,name="conv9_actvn")
            
            #layer 10
            conv10 = tf.layers.conv2d(inputs=conv9,filters=147,kernel_size=(11,11),padding="same",name="conv10",kernel_initializer=tf.truncated_normal_initializer)
            conv10 = tf.contrib.layers.instance_norm(inputs=conv10)
            conv10 = tf.nn.selu(conv10,name="conv10_actvn")
            
            #Image generation layer 
            outH = tf.layers.conv2d(inputs=conv10,filters=3,kernel_size=(11,11),padding="same",name="output",kernel_initializer=tf.truncated_normal_initializer)
            return outH
        
        
    def train(self):      
        #with tf.device(self.gpu):
        self.outH=self.generateImage()
        loss= self.loss()
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learningRate)
        Trainables=optimizer.minimize(loss)
        valid_image_summary =tf.summary.image('test_image_output',self.outH)
        loss_summary = tf.summary.scalar('Loss',loss)
        iters=0
        self.saver = tf.train.Saver()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        train_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,'train'),self.sess.graph)
        test_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,'test'),self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        process = psutil.Process(os.getpid())
        while not self.dataObj.epoch == self.maxEpoch :
            iters+=1
            inp,gt = self.dataObj.nextTrainBatch()
            loss = self.loss()
            t1=time.time()
            _,lval,t_summaries = self.sess.run([Trainables,loss,loss_summary], feed_dict= {self.depth : inp,self.rgb : gt})
            train_summary_writer.add_summary(t_summaries,iters)
            t2=time.time()         
            
            
            print(process.memory_info().rss/100000.)
            if not iters % self.print_freq:
                info = "Model Hallucinatore_s{} Epoch  {} : Iteration : {}/{} loss value : {:0.4f} Time per batch : {:0.3f}s \n".format(self.scale,self.dataObj.epoch,iters,(self.dataObj.epoch+1)*int(self.dataObj.dataLength/self.dataObj.batchSize),lval,t2-t1)
                print (info)   
                self.logger.write(info)
            if not iters % self.save_freq:
                info="Model Saved at iteration {}\n".format(iters)
                print (info)
                self.logger.write(info)
                self.saveModel(iters)
                
            if not iters % self.val_freq :
                val_inp,val_gt  = self.dataObj.nextTestBatch()
                val_loss,v_summaries,v_img_summaries = self.sess.run([loss,loss_summary,valid_image_summary],feed_dict={self.depth : val_inp,self.rgb : val_gt})
                test_summary_writer.add_summary(v_summaries, iters)
                test_summary_writer.add_summary(v_img_summaries,iters)
                info = "Validation Loss at iteration{} : {}\n".format(iters, val_loss)
                print (info)
                self.logger.write(info)
        print ("Training done ")
        self.logger.close()
    
    def testAll(self):
        self.restoreModel()
        loss=[]
        while not self.dataObj.test_epoch == 1 :
            x,y = self.dataObj.nextTestBatch()
            l = self.sess.run(self.loss(),feed_dict= {self.depth : x, self.rgb :y})
            loss.append(l)
        return np.mean(loss)
        
    def getHallucinatedImages(self,image_list):
        with tf.device(self.gpu):
            self.restoreModel()
            img_processed= self.dataObj.preProcessImages(image_list)
            generator = self.generateImage()
            output = self.sess.run(generator,{self.depth:img_processed})
            return output
    
    def loss(self):
        return tf.reduce_mean(2*tf.nn.l2_loss(self.outH-self.rgb))/(self.imageHeight*self.imageWidth*self.dataObj.batchSize*self.channels)   
    def smoothing_loss(self):
        I_Hgrad = tf.images
    def saveModel(self,iters):
        if not os.path.exists (self.modelLocation):
            os.makedirs(self.modelLocation)
        self.saver.save(self.sess,os.path.join(self.modelLocation,self.modelName),global_step = iters)
        
    def restoreModel(self):
        print (self.modelLocation)
        if not self.sess is None:
            if self.sess._opened :
                self.sess.close()
        sess=tf.Session()
        self.outH=self.generateImage()
        sav=tf.train.Saver()
        sav.restore(sess,self.restoreModelPath)
        self.sess = sess


# In[11]:

if __name__ == '__main__':
    sys.argv=['RGBHallucinator','8','0']
    parser =  argparse.ArgumentParser(description = "Mention the scale and GPU parameters")
    parser.add_argument('scale',type = int ,help = " What scale do you want to train it at ")
    parser.add_argument('gpu',type = int,help = " Which GPU do you want to train it in ")
    args= parser.parse_args()
    scale=args.scale
    gpu  =args.gpu
    tf.reset_default_graph()
    H = Hallucinator("config.ini",scale,gpu)
    H.testAll()


# In[ ]:

# import matplotlib.pyplot as plt
# %matplotlib inline
# h=Hallucinator("config.ini",1,0)
# inp,gt = h.dataObj.nextTestBatch()

# edge= tf.image.sobel_edges(h.rgb)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# out,edge = sess.run([h.rgb,edge],feed_dict = {h.rgb:gt})
# sess.close()
# plt.imshow(out[0])


# In[ ]:

# print edge.shape


# In[ ]:

# a=edge[0]


# In[ ]:

# y = a[:,:,:,0]


# In[ ]:

# y.shape


# In[ ]:

# plt.imshow(y)


# In[ ]:

# x = a[:,:,:,1]
# plt.imshow(x)


# In[ ]:



