#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
print (tf.__version__)
import Dataset
import time
import os
import configparser as ConfigParser
import sys
import argparse 
import psutil
#import  resnet18_linknet as ln
from linknet import linknet
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class Hallucinator ():    
    def __init__ (self,config_file,scale,gpu_num):
        print ("Initializing Hallucinator class")
        self.scale=scale
        self.gpu ="/gpu:{}".format(gpu_num)
        self.readConfiguration(config_file)
        self.inputs=tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='depthInput')             
        self.phase=tf.placeholder(tf.bool)
        self.output  =tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='grountTruth')
        self.dataObj = Dataset.dataReader(self.dataArguments)
        if not os.path.exists(self.summary_writer_dir):
            os.makedirs(self.summary_writer_dir)
        if not os.path.exists(self.modelLocation):
            os.makedirs(self.modelLocation)
        logPath = os.path.join(self.logDir,self.modelName)
        if not os.path.exists(logPath):
            os.makedirs(logPath)
        self.logDir = os.path.join(logPath,'log.txt')
        try :
            os.remove(self.logDir)
        except :
            pass
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        self.teacherScope = "Teacher"
        self.generatorScope = "Generator"
        self.sess = None
        with open(config_file) as fh :
            self.confInfo = fh.read()
        if self.model_choice == "linkNet":
            with tf.variable_scope(self.generatorScope):
                #linkNet = ln.LinkNet_resnt18(self.inputs, is_training=self.phase,num_classes =self.channels)
                self.model, self.end_points = linknet(self.inputs,3,None,self.phase)#linkNet.build_model()
            with tf.variable_scope(self.teacherScope) : 
                #teacherNet = ln.LinkNet_resnt18(self.inputs, is_training=self.phase,num_classes =self.channels)
                self.teacherModel , self.teacherEndpoints =linknet(self.inputs,3,None,self.phase) #teacherNet.build_model()
                
        if self.model_choice == "APG" :
            self.model = self.generateImage()
            
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
        self.colorLossType= config.get('DATA','colorLossType')
        self.corruptionLevel =float(config.get('DATA','corruptionLevel'))
        self.dataArguments = {"imageWidth":self.imageWidth,"imageHeight":self.imageHeight,"channels" : self.channels, "batchSize":self.batchSize,"train_file":self.train_file,"test_file":self.test_file,"scale":self.scale,"colorSpace":self.colorLossType,"corruptionLevel":self.corruptionLevel}    
        self.maxEpoch=int(config.get('TRAIN','maxEpoch'))
        self.generatorLearningRate = float(config.get('TRAIN','generatorLearningRate'))
        self.teacherLearningRate =  float (config.get('TRAIN','teacherLearningRate'))
        self.huberDelta  = float(config.get('TRAIN','huberDelta'))
        self.lambda1     = float(config.get('TRAIN','rmse_lambda'))
        self.lambda2     = float(config.get('TRAIN','smooth_lambda'))
        self.lambda3     = float(config.get('TRAIN','colorLoss_lambda'))
        self.model_choice = config.get('TRAIN','model')
        
        
        if int(config.get('TRAIN','activation')) == 0:
            self.activation = tf.nn.relu
            print("Relu activation has been chosen")
        else :
            self.activation = tf.nn.selu
        self.normType = config.get('TRAIN',"normalizationType")
        print (self.normType + " Normalization has been chosen")
        self.print_freq=int(config.get('LOG','print_freq'))
        self.save_freq = int (config.get('LOG','save_freq'))
        self.val_freq = int (config.get('LOG','val_freq'))
        self.modelName = config.get('LOG','modelName') +"_s{}".format(self.scale)
        self.modelLocation = config.get('LOG','modelLocation')
        self.modelLocation = os.path.join(self.modelLocation , self.modelName)
        self.checkPoint =  bool(int(config.get('LOG','checkpoint')))
        #self.restoreModelPath =config.get('LOG','restoreModelPath')
        self.logDir = config.get('LOG','logFile')
        self.summary_writer_dir =os.path.join(config.get('LOG','summary_writer_dir') ,self.modelName)    
        
        
    def normalization(self,feat,typeN):
        if typeN == "BATCH":
            return tf.layers.batch_normalization(feat, training = self.phase)
        if typeN == "INSTANCE":
            return tf.contrib.layers.instance_norm(feat)
            

    def generateImage(self):
        with tf.variable_scope(self.modelName, reuse=tf.AUTO_REUSE):
            mu,sigma=0,0.1
            strides=[1,1,1,1]
            #Layer 1 
            w1=tf.Variable(tf.truncated_normal(shape=(11,11,3,147),mean=mu,stddev= sigma))
            b1=tf.Variable(tf.zeros(147))
            conv1=tf.nn.conv2d(self.inputs,w1,strides,padding='SAME')
            conv1=tf.nn.bias_add(conv1,b1)
            conv1=self.normalization(conv1,self.normType)
            conv1=self.activation(conv1)
            #chk1=tf.check_numerics(conv1,"conv1")
            #Layer 2
            w2=tf.Variable(tf.truncated_normal(shape=(11,11,147,36),mean=mu,stddev=sigma))
            b2=tf.Variable(tf.zeros(36))
            conv2=tf.nn.conv2d(conv1,w2,strides,padding='SAME')
            conv2=tf.nn.bias_add(conv2,b2)
            conv2=self.normalization(conv2,self.normType)
            conv2=self.activation(conv2)
            #Layer 3
            w3=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b3=tf.Variable(tf.zeros(36))
            conv3=tf.nn.conv2d(conv2,w3,strides,padding='SAME')
            conv3=tf.nn.bias_add(conv3,b3)
            conv3=self.normalization(conv3,self.normType)
            conv3=self.activation(conv3)
            #Layer 4
            w4=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b4=tf.Variable(tf.zeros(36))
            conv4=tf.nn.conv2d(conv3,w4,strides,padding='SAME')
            conv4=tf.nn.bias_add(conv4,b4)
            conv4=self.normalization(conv4,self.normType)
            conv4=self.activation(conv4)
            #Layer 5
            w5=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b5=tf.Variable(tf.zeros(36))
            conv5=tf.nn.conv2d(conv4,w5,strides,padding='SAME')
            conv5=tf.nn.bias_add(conv5,b5)
            conv5=self.normalization(conv5,self.normType)
            conv5=self.activation(conv5)
            #Layer 6
            w6=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b6=tf.Variable(tf.zeros(36))
            conv6=tf.nn.conv2d(conv5,w6,strides,padding='SAME')
            conv6=tf.nn.bias_add(conv6,b6)
            conv6=self.normalization(conv6,self.normType)
            conv6=self.activation(conv6)
            #Layer 7
            w7=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b7=tf.Variable(tf.zeros(36))
            conv7 =tf.nn.conv2d(conv6,w7,strides,padding='SAME')
            conv7=tf.nn.bias_add(conv7,b7)
            conv7=self.normalization(conv7,self.normType)
            conv7=self.activation(conv7)
            #layer 8
            w8=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b8=tf.Variable(tf.zeros(36))
            conv8 = tf.nn.conv2d(conv7,w8,strides,padding='SAME')
            conv8 = tf.nn.bias_add(conv8,b8)
            conv8 = self.normalization(conv8,self.normType)
            conv8 = self.activation(conv8)
            #layer 9
            w9=tf.Variable(tf.truncated_normal(shape=(11,11,36,36),mean=mu,stddev=sigma))
            b9=tf.Variable(tf.zeros(36))
            conv9 = tf.nn.conv2d(conv8,w9,strides,padding='SAME')
            conv9 = tf.nn.bias_add(conv9,b9)
            conv9 = self.normalization(conv9,self.normType)
            conv9 = self.activation(conv9)
            #layer 10
            w10=tf.Variable(tf.truncated_normal(shape=(11,11,36,147),mean=mu,stddev=sigma))
            b10=tf.Variable(tf.zeros(147))
            conv10 =tf.nn.conv2d(conv9,w10,strides,padding='SAME')
            conv10 = tf.nn.bias_add(conv10,b10)
            conv10 = self.normalization(conv10,self.normType)
            conv10 = self.activation(conv10)
            #layer 11
            w11=tf.Variable(tf.truncated_normal(shape=(11,11,147,3),mean=mu,stddev=sigma))
            b11=tf.Variable(tf.zeros(3))
            conv11 = tf.nn.conv2d(conv10,w11,strides,padding='SAME')
            conv11 = tf.nn.bias_add(conv11,b11,name="output")
            return conv11
    
    def train(self,trainModel): 
        if os.path.isfile (self.logDir):
            print ("This log file with same name exisits. Likely that the model file exists. If you want to cancel, cancel in the next 5 seconds")
            time.sleep(5)
            self.logger = open(self.logDir,'a')
        else :
            self.logger = open(self.logDir,'w')
            self.logger.write(trainModel + " Training")
            self.logger.write("***********************")
            self.logger.write(self.confInfo +'\n\n\n')
        
        if trainModel == 'TEACHER':
            print ("The teacher is being trained right now ")
            self.outH=self.teacherModel
            loss= self.autoEncoderLoss()
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = self.teacherScope) 
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = self.teacherScope)
            learningRate = self.teacherLearningRate
            corruptionFlag = True
            
        elif trainModel == 'GENERATOR':
            print ("The Generator model is being trained right now")
            self.outH = self.model
            loss = self.loss()
            variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = self.generatorScope)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = self.generatorScope) 
            learningRate = self.generatorLearningRate
            corruptionFlag = False
        
        else :
            print ("Invalid model choice")
            return None
    
        optimizer=tf.train.AdamOptimizer(learning_rate=learningRate)
        with tf.control_dependencies(update_ops): 
            Trainables=optimizer.minimize(loss,var_list = variables)
        valid_image_summary =tf.summary.image('test_image_output',self.outH)
        loss_summary = tf.summary.scalar('Loss',loss)
        iters=0
        
        self.saver = tf.train.Saver() 
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        train_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,trainModel+'train'),self.sess.graph)
        test_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,trainModel+'test'),self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        process = psutil.Process(os.getpid())
        self.dataObj.resetTrainBatch()
        while not self.dataObj.epoch == self.maxEpoch :
            #iters+=1
            inp,gt = self.dataObj.nextTrainBatch(corruptionFlag=corruptionFlag)
            t1=time.time()
            if trainModel == 'TEACHER' :
                lval= self.sess.run(loss, feed_dict= {self.inputs : gt,self.phase : True})
            if trainModel == 'GENERATOR':
                lval= self.sess.run(loss, feed_dict= {self.inputs : inp,self.phase : True ,self.output : gt})
            _,lval,t_summaries = self.sess.run([Trainables,loss,loss_summary], feed_dict= {self.inputs : inp,self.phase : True ,self.output : gt})
            train_summary_writer.add_summary(t_summaries,iters)
            t2=time.time()      
            if not iters % self.print_freq:
                info = "Model Hallucinator_s{} Epoch  {} : Iteration : {}/{} loss value : {:0.4f} \n".format(self.scale,self.dataObj.epoch,iters,(self.maxEpoch)*int(self.dataObj.dataLength/self.dataObj.batchSize),lval) +"Memory used : {:0.4f} GB Time per batch : {:0.3f}s Model : {} \n".format(process.memory_info().rss/100000000.,t2-t1,self.modelName) 
                print (info)   
                self.logger.write(info)
                
            if not iters % self.save_freq:
                info="Model Saved at iteration {}\n".format(iters)
                print (info)
                self.logger.write(info)
                self.saveModelAll(iters)
                
            if not iters % self.val_freq :
                val_inp,val_gt  = self.dataObj.nextTestBatch(corruptionFlag=corruptionFlag)
                if trainModel == "TEACHER":
                    val_loss,v_summaries,v_img_summaries = self.sess.run([loss,loss_summary,valid_image_summary],feed_dict={self.inputs : val_gt,self.phase:False})
                if trainModel == "GENERATOR":
                    val_loss,v_summaries,v_img_summaries = self.sess.run([loss,loss_summary,valid_image_summary],feed_dict={self.inputs :val_inp ,self.output: val_gt,self.phase:False})
                test_summary_writer.add_summary(v_summaries, iters)
                test_summary_writer.add_summary(v_img_summaries,iters)
                info = "Validation Loss at iteration{} : {}\n".format(iters, val_loss)
                print (info)
                self.logger.write(info)
            iters+=1
        print ("Training done ")
        self.saveModelAll(iters)
        self.logger.close()
    
    def testAll(self,modelChoice ):
        self.dataObj.resetTestBatch()
        #self.restoreModel(modelChoice)
        self.resotreModelAll()
        loss=[]
        if modelChoice == "TEACHER":
            loss_func = self.autoEncoderLoss()
            corruptionFlag =  True
        elif modelChoice == "GENERATOR" :
            loss_func = self.loss()
            corruptionFlag = False
        else : 
            print ("INVALID MODEL CHOICE !!!")
            return None
        print ("Testing")
        while not self.dataObj.test_epoch == 1 :
            x,y = self.dataObj.nextTestBatch(corruptionFlag=corruptionFlag)
            
            if modelChoice == 'TEACHER' :
                print ("---------------------------------------")
                lval= self.sess.run(loss_func, feed_dict= {self.inputs : y,self.phase : False})
            if modelChoice == 'GENERATOR':
                lval= self.sess.run(loss_func, feed_dict= {self.inputs : x ,self.phase : False ,self.output : y})
            loss.append(lval)
            print("Test start : {} Model: {} Test Loss : {}".format(self.dataObj.test_start,self.modelName,lval))
        return np.mean(loss)
        
    def getHallucinatedImages(self,image_list):
        self.restoreModel()
        img_processed= self.dataObj.loadImages(image_list,False)
        output = self.sess.run(self.outH,{self.inputs:img_processed,self.phase : False})
        output = self.dataObj.postProcessImages(output)
        return output
    
    def getTeacherImages(self,image_list):
        self.restoreModel("TEACHER")
        img_processed= self.dataObj.loadImages(image_list,True)
        inp,gt = self.dataObj.nextTrainBatch()
        output = self.sess.run(self.outH,{self.inputs:gt,self.phase : False})
        output = self.dataObj.postProcessImages(output)
        return output,gt
        
    
    def loss (self) :
        return self.lambda1*self.l2_loss() + self.lambda2*self.smoothing_loss() 

    def l2_loss(self):
        return tf.sqrt(tf.losses.mean_squared_error(self.output,self.outH))  
    
    def autoEncoderLoss(self):
        return tf.sqrt(tf.losses.mean_squared_error(self.inputs,self.outH))  
    
    def smoothing_loss(self):
        I_Hgrad    = tf.image.sobel_edges(self.outH)
        I_Hedge    = I_Hgrad[:,:,:,:,0] + I_Hgrad[:,:,:,:,1]
        zeros      = tf.zeros_like(I_Hedge)
        I_Hhuber   = tf.losses.huber_loss(I_Hedge,zeros,delta = self.huberDelta,reduction=tf.losses.Reduction.NONE)
            
        I_RGBgrad  = tf.image.sobel_edges(self.output)
        I_RGBedge  = I_RGBgrad[:,:,:,:,0] + I_RGBgrad[:,:,:,:,1]
        I_RGBhuber = tf.losses.huber_loss(I_RGBedge,zeros,delta = self.huberDelta, reduction=tf.losses.Reduction.NONE)

        edge_aware_weight   = tf.exp(-1*I_RGBhuber)
        weighted_smooth_img = tf.multiply(I_Hhuber, edge_aware_weight)
        loss_val = tf.reduce_mean(weighted_smooth_img) 
        return loss_val
   
    def saveModel(self,iters,modelChoice):
        path = os.path.join(self.modelLocation,modelChoice)
        if not os.path.exists (path):
            os.makedirs(path)
        self.saver.save(self.sess,os.path.join(path,modelChoice+'_'+self.modelName),global_step = iters)
        
    def saveModelAll(self,iters):
        path = self.modelLocation
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess,os.path.join(path,self.modelName),global_step=iters)
        
    def restoreModelAll(self):
        path = self.modelLocation
        if self.checkPoint:
            print ("Using the latest trained model in check point file")
            self.restoreModelPath = tf.train.latest_checkpoint(path)
            print (" Model at {} restored".format(self.restoreModelPath))
        else : 
            self.restoreModelPath = config.get('LOG','restoreModelPath')
            self.restoreModelPath = self.restoreModelPath
            
        if not self.sess is None:
            if self.sess._opened :
                self.sess.close()
        sess=tf.Session()
        self.outH = self.teacherModel
        outH = self.model
            
        saver=tf.train.Saver()
        saver.restore(sess,self.restoreModelPath)
        self.sess=sess
                
    def restoreModel(self,modelChoice):
        path = os.path.join(self.modelLocation,modelChoice)
        if self.checkPoint:
            print ("Using the latest trained model in check point file")
            self.restoreModelPath = tf.train.latest_checkpoint(path)
            print (" Model at {} restored".format(self.restoreModelPath))
        else : 
            self.restoreModelPath = config.get('LOG','restoreModelPath')
            self.restoreModelPath = os.path.join(self.restoreModelPath,modelChoice)
            
        if not self.sess is None:
            if self.sess._opened :
                self.sess.close()
                
        sess=tf.Session()
        if modelChoice == "TEACHER":
            print ("Restoring Teacher model")
            self.outH = self.teacherModel
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = self.teacherScope) 
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = self.teacherScope)
        elif modelChoice == "GENERATOR":
            print ("Restoring Generator model")
            self.outH = self.model
            variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = self.generatorScope)
        else :
            print ("INVALID MODEL CHOICE")
            return None
        
        saver=tf.train.Saver(var_list = variables)
        saver.restore(sess,self.restoreModelPath)
        self.sess=sess
        
        
    def layerWise (self) :
        sliceBegin = self.teacherEndpoints['encode1']
        sliceEnd = self.teacherEndpoints['decode1']
    
        


# In[5]:





