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
from linknet import *
import matplotlib.pyplot as plt
from Architecture import architecture


# In[2]:


class Hallucinator ():    
    def __init__ (self,config_file,scale,gpu_num):
        print ("Initializing Hallucinator class")
        tf.reset_default_graph()
        self.scale=scale
        self.gpu ="/gpu:{}".format(gpu_num)
        self.readConfiguration(config_file)
        self.inputs=tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='Input')             
        self.phase=tf.placeholder(tf.bool,name = "phase")
        self.output  =tf.placeholder(tf.float32,(None,self.imageHeight,self.imageWidth,self.channels),name='grountTruth')
        self.dataObj = Dataset.dataReader(self.dataArguments)
        
        self.continue_training = False
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
        
        
        self.teacherScope = "Teacher"
        self.generatorScope = "Generator"
        self.sess = None
        with open(config_file) as fh :
            self.confInfo = fh.read()        
        self.generatorModel =None
        if self.model_choice == "APG" :
            self.model = self.generateImage()
        elif self.model_choice == "linknet":
            self.filters = [64, 128, 256, 512]
            self.filters_m = [64, 128, 256, 512][::-1]
            self.filters_n = [64, 64, 128, 256][::-1]
            self.generatorModel = lambda inp: linknet(inp,num_classes =3,reuse = None,is_training = self.phase)
        elif self.model_choice == "aggregated":
            print ("Trying out new architecture")
            arch= architecture([[(3,3),(11,11)],[(5,5),(7,7)],[(11,11),(9,9)]],[(3,3),(7,7),(11,11)])
            self.generatorModel = lambda inp : arch.buildNetwork_base2(inp,self.phase,"Hallucinator")
            
    def readConfiguration(self,config_file):
        print ("Reading configuration File")
        config = ConfigParser.ConfigParser()
        config.read(config_file)
        self.imageWidth=int(int(config.get('DATA','imageWidth'))/self.scale)
        self.imageHeight=int(int(config.get('DATA','imageHeight'))/self.scale)
        self.channels=int (config.get('DATA','channels'))
        self.train_file = config.get ('DATA','train_file')
        self.test_file  = config.get ('DATA','test_file')
        autoencoder_train_file = config.get('DATA','autoencoder_train_file')
        autoencoder_test_file  = config.get('DATA','autoencoder_test_file')
        self.batchSize  = int(config.get ('DATA','batchSize'))
        self.colorLossType= config.get('DATA','colorLossType')
        self.corruptionLevel =float(config.get('DATA','corruptionLevel'))
        self.dataArguments = {"imageWidth":self.imageWidth,"imageHeight":self.imageHeight,"channels" : self.channels, "batchSize":self.batchSize,"train_file":self.train_file,"test_file":self.test_file,"scale":self.scale,"colorSpace":self.colorLossType,"corruptionLevel":self.corruptionLevel,"autoencoder_train_file": autoencoder_train_file,"autoencoder_test_file": autoencoder_test_file}    
        self.maxEpoch=int(config.get('TRAIN','maxEpoch'))
        self.generatorLearningRate = float(config.get('TRAIN','generatorLearningRate'))
        self.teacherLearningRate =  float (config.get('TRAIN','teacherLearningRate'))
        self.huberDelta  = float(config.get('TRAIN','huberDelta'))
        self.lambda1     = float(config.get('TRAIN','rmse_lambda'))
        self.lambda2     = float(config.get('TRAIN','smooth_lambda'))
        self.lambda3     = float(config.get('TRAIN','colorLoss_lambda'))
        self.model_choice = config.get('TRAIN','model')
        self.dropout_probability   = config.get('TRAIN', 'dropout')
        self.restore_name = config.get('LOG','restoreModelName')
        self.restoreModelPath = config.get('LOG','restoreModelPath')
        self.num_gpus = int(config.get('TRAIN','num_gpus'))
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
    
    def average_gradients(self,tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads

    def initModel(self,trainModel):
        if trainModel == 'GENERATOR':
                print ("The Generator model is being initialized right now")
                self.learningRate = self.generatorLearningRate
                self.dataGrabber = self.dataObj.nextTrainBatch
                self.dataGrabberTest = self.dataObj.nextTestBatch
                self.scope = self.generatorScope
                self.model = self.generatorModel
                return True
                
        else :
            print ("Invalid model choice")
            
    def train(self,trainModel): 
        with tf.device ('/cpu:0'):
            self.logger = open(self.logDir,'w')
            self.logger.write(trainModel + " Training")
            self.logger.write("***********************")
            self.logger.write(self.confInfo +'\n\n\n')
            self.dataObj.resetTrainBatch()
            self.dataObj.resetTestBatch()
            process = psutil.Process(os.getpid())
            ret = self.initModel(trainModel)
            if not ret :
                print ("Invalid model choice")
                return None
            gradientsList = []
            lossList = []
            perGPUBatch = self.batchSize//self.num_gpus
            for gpu_n in range(self.num_gpus):
                with tf.device('/gpu:%d' % gpu_n):
                    with tf.variable_scope(self.scope,reuse= tf.AUTO_REUSE):
                        _inp = self.inputs[gpu_n * perGPUBatch: (gpu_n+1) *perGPUBatch]
                        _out = self.output[gpu_n * perGPUBatch: (gpu_n+1) *perGPUBatch]
                        outH = self.model(_inp)
                        loss = self.loss(_pred =outH,_gt = _out)
                        lossList.append(loss)
                        optimizer=tf.train.AdamOptimizer(learning_rate=self.learningRate)
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops): 
                            gradients=optimizer.compute_gradients(loss)
                        if gpu_n == 0 :
                            valid_image_summary =tf.summary.image('test_image_output',outH)
                            loss_summary = tf.summary.scalar('Loss',loss)
                        gradientsList.append(gradients)
            averagedLoss = tf.reduce_mean(lossList)
            averagedGradients = self.average_gradients(gradientsList)
            Trainables = optimizer.apply_gradients(averagedGradients)
            iters=0
            self.saver = tf.train.Saver() 
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement = True
            #config.log_device_placement = True
            self.sess = tf.Session(config=config)
            train_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,trainModel+'train'),self.sess.graph)
            test_summary_writer=tf.summary.FileWriter(os.path.join(self.summary_writer_dir,trainModel+'test'),self.sess.graph)     
            self.sess.run(tf.global_variables_initializer())
            while not self.dataObj.epoch == self.maxEpoch :
                t1=time.time()
                inp,gt =self.dataGrabber(False)
                _,lval,t_summaries= self.sess.run([Trainables,averagedLoss,loss_summary], feed_dict= {self.inputs : inp,self.output : gt,self.phase : True})
                t2=time.time()    
                
                if not iters % self.print_freq:
                    info = "Model Hallucinator_s{} Epoch  {} : Iteration : {}/{} loss value : {:0.4f} \n".format(self.scale,self.dataObj.epoch,iters,(self.maxEpoch)*int(self.dataObj.dataLength/self.dataObj.batchSize),lval) +"Memory used : {:0.4f} GB Time per batch : {:0.3f}s Model : {} \n".format(process.memory_info().rss/100000000.,t2-t1,self.modelName) 
                    print (info)   
                    self.logger.write(info)
                    train_summary_writer.add_summary(t_summaries,iters)

                if not iters % self.save_freq:
                    info="Model Saved at iteration {}\n".format(iters)
                    print (info)
                    self.logger.write(info)
                    self.saveModel(iters,trainModel)

                if not iters % self.val_freq :           
                    val_inp,val_gt  = self.dataGrabberTest()
                    if trainModel == "TEACHER":
                        val_inp = np.copy(val_gt)
    #                 import matplotlib.pyplot as plt 
    #                 plt.imshow(np.uint8(val_inp[0]))
    #                 plt.figure()
    #                 plt.imshow(np.uint8(val_gt[0]))
    #                 s
                    val_loss,v_summaries,v_img_summaries = self.sess.run([averagedLoss,loss_summary,valid_image_summary],feed_dict={self.inputs : val_inp,self.output : val_gt,self.phase:False})  
                    test_summary_writer.add_summary(v_summaries, iters)
                    test_summary_writer.add_summary(v_img_summaries,iters)
                    info = "Validation Loss at iteration{} : {}\n".format(iters, val_loss)
                    print (info)
                    self.logger.write(info)
                iters+=1
            print ("Training done ")
            self.saveModel(iters,trainModel)
            self.logger.close()
    
    def testAll(self,modelChoice ):
        self.dataObj.resetTestBatch()
        outH = self.restoreModel(modelChoice)
        loss=[]
        loss_func = tf.reduce_mean(tf.abs(outH - self.output))
        print ("Testing")
        while not self.dataObj.test_epoch == 1 :
            x,y = self.dataGrabberTest()
            lval= self.sess.run(loss_func, feed_dict= {self.inputs : x ,self.phase : False ,self.output : y})
            loss.append(lval)
            print("Test start : {} Model: {} Test Loss : {}".format(self.dataObj.test_start,self.modelName,lval))
        return np.mean(loss)
        
    def getHallucinatedImages(self,image_list):
        if self.sess is None:
            self.restoreModel("GENERATOR")
        img_processed= self.dataObj.loadImages(image_list,False)
#         img_processed = cv2.imread(image_list[0],-1)
#         img_processed = np.uint8((img_processed/np.max(img_processed))*255.)
#         img_processed = cv2.resize(np.stack([img_processed,img_processed,img_processed],axis=-1),(640,480))
#         plt.imshow(img_processed)
#         img_processed = np.float32([img_processed])   
#         img_processed = img_processed.reshape(-1,480,640,3)
        output = self.sess.run(self.outH,{self.inputs:img_processed,self.phase : False})
        output = self.dataObj.postProcessImages(output)
        return output
    
    def getTeacherImages(self,image_list):
        self.restoreModel("TEACHER")
        img_processed= self.dataObj.loadImages(image_list)
        #inp,gt = self.dataObj.nextTrainBatch()
        output = self.sess.run(self.outH,{self.inputs:img_processed,self.phase : False})
        output = self.dataObj.postProcessImages(output)
        return output
        
    def getReconstructedImage(self,image):
        img = cv2.imread(image)
        if img is None :
            print ("Not valid image")
            return None
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img = cv2.resize(img,(640,480))
        img = self.dataObj.preProcessImages([img])
        if self.sess is None :
            self.restoreModel("BOTH")
        output = self.sess.run(self.stage1,{self.inputs:img,self.phase : False})
        output = self.sess.run(self.stage2,{self.inputs:output,self.phase : False})
        return self.dataObj.postProcessImages(output)

    def loss (self,_pred,_gt) :
        with tf.variable_scope("Final_loss"):
            return self.lambda1*self.l2_loss(_pred,_gt) + self.lambda2*self.smoothing_loss(_pred,_gt) 

    def l2_loss(self,_pred,_gt):
        with tf.variable_scope ("RMSE_loss"):
            return tf.sqrt(tf.losses.mean_squared_error(_pred,_gt))  
   
    def smoothing_loss(self,_pred,_gt):
        with tf.variable_scope ("smoothing_loss") :
            I_Hgrad    = tf.image.sobel_edges(_pred)
            I_Hedge    = I_Hgrad[:,:,:,:,0] + I_Hgrad[:,:,:,:,1]
            zeros      = tf.zeros_like(I_Hedge)
            I_Hhuber   = tf.losses.huber_loss(I_Hedge,zeros,delta = self.huberDelta,reduction=tf.losses.Reduction.NONE)

            I_RGBgrad  = tf.image.sobel_edges(_gt)
            I_RGBedge  = I_RGBgrad[:,:,:,:,0] + I_RGBgrad[:,:,:,:,1]
            I_RGBhuber = tf.losses.huber_loss(I_RGBedge,zeros,delta = self.huberDelta, reduction=tf.losses.Reduction.NONE)
            edge_aware_weight   = tf.exp(-1*I_RGBhuber)
            weighted_smooth_img = tf.multiply(I_Hhuber, edge_aware_weight)
            loss_val = tf.reduce_mean(weighted_smooth_img) 
        return loss_val

                       
    def featureSpaceLoss (self):
        with tf.variable_scope("Featurespace_loss"):
            feature_loss =            rmse("encode1") + rmse("encode2") + rmse("encode3") + rmse("encode4") +            rmse ("decode1") + rmse("decode2") + rmse("decode3") + rmse("decode4")
        return feature_loss
            
    def saveModel(self,iters,modelChoice):
        path = os.path.join(self.modelLocation,modelChoice)
        if not os.path.exists (path):
            os.makedirs(path)
        self.saver.save(self.sess,os.path.join(path,modelChoice+'_'+self.modelName),global_step = iters)
                    
    def restoreModel(self,modelChoice):
        path = os.path.join(self.modelLocation,modelChoice)
        if self.checkPoint:
            print ("Using the latest trained model in check point file")
            self.restoreModelPath = tf.train.latest_checkpoint(path)
            print (" Model at {} restored".format(self.restoreModelPath))
        else : 
            
            self.restoreModelPath = os.path.join(self.restoreModelPath,modelChoice,modelChoice+'_'+self.restore_name)
            
        if not self.sess is None:
            if self.sess._opened :
                self.sess.close()
                
        sess=tf.Session()
        ret = self.initModel(modelChoice)
        if not ret :
            print ("Invalid model Choice")
            return None
        with tf.variable_scope(self.scope):
            outH =  self.model(self.inputs)
        variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = self.scope)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = self.scope)
        saver=tf.train.Saver(var_list = variables)
        saver.restore(sess,self.restoreModelPath)
        self.sess=sess
        return outH