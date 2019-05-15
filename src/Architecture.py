import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
from collections import defaultdict

class architecture():
    def __init__(self,encoder_kernels,decoder_kernels):
        self.ENCODER_KERNELS = encoder_kernels
        self.DECODER_KERNELS = decoder_kernels
        
    def aggregatedConvolutionBN(self,x,filters,isTraining,kernelList=[[(3,3),(5,5),(7,7)]],scope= "AggregatedConv",activation= None,subSample=0):
        with tf.variable_scope(scope):
            featureMapList = []
            dilationRate =0
            for kernels in kernelList :
                dilationRate +=1
                for kernel in kernels:
                    featureMapList.append(tf.layers.conv2d(inputs=x,filters =filters, kernel_size = kernel,padding = 'same',activation = None,dilation_rate = dilationRate))    

            featureMap  = tf.concat (featureMapList,axis =-1) 
            featureMap  = tf.layers.batch_normalization(featureMap,training = isTraining)
            if not activation is None :

                featureMap = activation(featureMap)
            if subSample :
                featureMap = tf.layers.conv2d(inputs = featureMap,filters = featureMap.shape[-1],kernel_size = (3,3),padding = 'same',activation = activation,strides=subSample)  
            return featureMap
        
    def aggregatedTransposedConvolutionBN(self,x,filters,isTraining,kernelList=[(3,3),(7,7),(11,11)],scope= "AggregatedTransposedConv",activation= None,superSample=0):
        with tf.variable_scope(scope):
            featureMapList = []
            for kernel in kernelList:
                featureMapList.append(tf.layers.conv2d_transpose(inputs=x,filters =filters, kernel_size = kernel,padding = 'same',activation = None,strides = superSample))    

            featureMap  = tf.concat (featureMapList,axis =-1) 
            featureMap  = tf.layers.batch_normalization(featureMap,training = isTraining)
            if not activation is None :
                featureMap = activation(featureMap)    
            return featureMap
    
    def EncoderBlockBase2 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedConvolutionBN(x,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv1_b2',activation = tf.nn.relu,subSample =2)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv2_b2',activation = None)   
            shortcut = tf.layers.conv2d(x,out.shape[-1],kernel_size= (3,3),padding = "same",strides= (2,2))
            out = tf.nn.relu(out+shortcut)
        return out
    def EncoderBlockBase4 ( self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedConvolutionBN(x,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv1_b4',activation = tf.nn.relu,subSample =4)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv2_b4',activation = tf.nn.relu)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv3_b4',activation = None)   
            shortcut = tf.layers.conv2d(x,out.shape[-1],kernel_size= (3,3),padding = "same",strides= (4,4))
            out = tf.nn.relu(out+shortcut)
        return out
    
    def EncoderBase4 ( self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedConvolutionBN(x,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv1_b4',activation = tf.nn.relu,subSample =4)               
            out = tf.layers.conv2d(out,out.shape[-1],kernel_size= (3,3),padding = "same",activation=tf.nn.relu)    
        return out
    
    def EncoderBase8 ( self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedConvolutionBN(x,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv1_b8',activation = tf.nn.relu,subSample =8)               
            out = tf.layers.conv2d(out,out.shape[-1],kernel_size= (3,3),padding = "same", activation = tf.nn.relu)  
        return out
    
    def EncoderBase16 ( self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedConvolutionBN(x,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'EncConv1_b16',activation = tf.nn.relu,subSample =16)               
            out = tf.layers.conv2d(out,out.shape[-1],kernel_size= (3,3),padding = "same",activation=tf.nn.relu)   
        return out
    
    def DecoderBlockBase2 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedTransposedConvolutionBN(x,filters,training,kernelList = self.DECODER_KERNELS,scope= "DecConv1_b2",activation = tf.nn.relu,superSample=2)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'DecConv2_b2',activation = None)   
            shortcut = tf.layers.conv2d_transpose(x,out.shape[-1],kernel_size= (3,3),padding = "same",strides= (2,2))
            out =tf.nn.relu(out + shortcut)
        return out
    
    def DecoderBlockBase4 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedTransposedConvolutionBN(x,filters,training,kernelList = self.DECODER_KERNELS,scope= "DecConv1_b4",activation = tf.nn.relu,superSample=4)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'DecConv2_b4',activation = tf.nn.relu)   
            out = self.aggregatedConvolutionBN(out,filters,training,kernelList =self.ENCODER_KERNELS,scope = 'DecConv3_b4',activation = None)   
            shortcut = tf.layers.conv2d_transpose(x,out.shape[-1],kernel_size= (3,3),padding = "same",strides= (4,4))
            out =tf.nn.relu(out + shortcut)
        return out
    
    def DecoderBase4 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedTransposedConvolutionBN(x,filters,training,kernelList = self.DECODER_KERNELS,scope= "DecConv1_b4",activation = tf.nn.relu,superSample=4)      
            out = tf.layers.conv2d_transpose(out,out.shape[-1],kernel_size= (3,3),padding = "same",activation=tf.nn.relu)  
        return out
    
    def DecoderBase8 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedTransposedConvolutionBN(x,filters,training,kernelList = self.DECODER_KERNELS,scope= "DecConv1_b8",activation = tf.nn.relu,superSample=8)       
            out = tf.layers.conv2d_transpose(out,out.shape[-1],kernel_size= (3,3),padding = "same",activation=tf.nn.relu)             
        return out

    def DecoderBase16 (self,x,filters,training,scope):
        with tf.variable_scope(scope):
            out = self.aggregatedTransposedConvolutionBN(x,filters,training,kernelList = self.DECODER_KERNELS,scope= "DecConv1_b16",activation = tf.nn.relu,superSample=16)       
            out = tf.layers.conv2d_transpose(out,out.shape[-1],kernel_size= (3,3),padding = "same",activation=tf.nn.relu)             
        return out
    
    def buildNetwork_base2(self,x,phase,scope):
        with tf.variable_scope(scope):
            enc1 = self.EncoderBlockBase2(   x,8,phase,"Encoder1_Base2")
            enc2 = self.EncoderBlockBase2(enc1,16,phase,"Encoder2_Base2")
            enc3 = self.EncoderBlockBase2(enc2,32,phase,"Encoder3_Base2")
            enc4 = self.EncoderBlockBase2(enc3,48,phase,"Encoder4_Base2")
            dec4 = self.DecoderBlockBase2(enc4,32,phase,"Decoder1_Base2")
            dec3 = self.DecoderBlockBase2(dec4 + enc3,16,phase,"Decoder2_Base2")
            dec2 = self.DecoderBlockBase2(dec3 + enc2,8,phase,"Decoder3_Base2")
            dec1 = self.DecoderBlockBase2(dec2 + enc1,8,phase,"Decoder4_Base2")
            logits =tf.layers.conv2d(dec1,filters=3,padding="same",activation = None,kernel_size = (5,5),name= "preFinal")
            logits =tf.layers.conv2d(logits,filters=3,padding = "same",activation = None,kernel_size = (3,3),name= "logits")
            return logits
    
    def buildNetwork_base4(self,x,phase,scope):
    
        with tf.variable_scope(scope):
            ''' Encoder structure'''
            enc1 = self.EncoderBlockBase2(   x,8,phase,"Encoder1_Base2")
            enc2a = self.EncoderBlockBase2(enc1,16,phase,"Encoder2_Base2")
            enc2b = self.EncoderBlockBase4(x,16,phase,"Encoder1_Base4")
            enc2 = tf.concat([enc2a,enc2b],axis =-1)
            enc3 = self.EncoderBlockBase2(enc2,32,phase,"Encoder3_Base2")
            enc4a = self.EncoderBlockBase2(enc3,32,phase,"Encoder4_Base2")
            enc4b = self.EncoderBlockBase4(enc2,32,phase,"Encoder2_Base4")
            enc4 = tf.concat([enc4a,enc4b],axis =-1)
            
            ''' Decoder structure'''
            dec4 = self.DecoderBlockBase2(enc4,32,phase,"Decoder1_Base2")
            dec3a = self.DecoderBlockBase2(dec4 + enc3,16,phase,"Decoder2_Base2")
            dec3b =  self.DecoderBlockBase4(enc4,16,phase,"Decoder1_Base4")
            dec3 = tf.concat([dec3a,dec3b],axis=-1)
            dec2 = self.DecoderBlockBase2(dec3 + enc2,8,phase,"Decoder3_Base2")
            dec1a = self.DecoderBlockBase2(dec2 + enc1,8,phase,"Decoder4_Base2")
            dec1b = self.DecoderBlockBase4(dec3,8,phase,"Decoder2_Base4")
            dec1 = tf.concat([dec1a,dec1b],axis = -1)
            logits =tf.layers.conv2d(dec1,filters=3,padding="same",activation = None,kernel_size = (5,5),name= "preFinal")
            logits =tf.layers.conv2d(logits,filters=3,padding = "same",activation = None,kernel_size = (3,3),name= "logits")
            return logits
    
    def buildNetwork_base2_3layers(self,x,phase,scope):
        with tf.variable_scope(scope):
            enc1 = self.EncoderBlockBase2(   x,8,phase,"Encoder1_Base2")
            enc2 = self.EncoderBlockBase2(enc1,16,phase,"Encoder2_Base2")
            enc3 = self.EncoderBlockBase2(enc2,32,phase,"Encoder3_Base2")
            dec3 = self.DecoderBlockBase2(enc3,16,phase,"Decoder2_Base2")
            dec2 = self.DecoderBlockBase2(dec3 + enc2,8,phase,"Decoder3_Base2")
            dec1 = self.DecoderBlockBase2(dec2 + enc1,8,phase,"Decoder4_Base2")
            logits =tf.layers.conv2d(dec1,filters=3,padding="same",activation = None,kernel_size = (5,5),name= "preFinal")
            logits =tf.layers.conv2d(logits,filters=3,padding = "same",activation = None,kernel_size = (3,3),name= "logits")
            return logits
        
    def buildNetwork_multibase(self,x,phase,scope):
    
        with tf.variable_scope(scope):
            ''' Encoder structure'''
            enc1 = self.EncoderBlockBase2(   x,8,phase,"Encoder1_Base2")
            enc2a = self.EncoderBlockBase2(enc1,16,phase,"Encoder2_Base2")
            enc2b = self.EncoderBase4(x,16,phase,"Encoder1_Base4")
            enc2 = tf.concat([enc2a,enc2b],axis =-1)
            enc3a = self.EncoderBlockBase2(enc2,32,phase,"Encoder3_Base2")
            enc3b = self.EncoderBase8(x,32,phase,"Encoder1_Base8")
            enc3 = tf.concat([enc3a,enc3b],axis = -1)
            enc4a = self.EncoderBlockBase2(enc3,32,phase,"Encoder4_Base2")
            enc4b = self.EncoderBase16(x,32,phase,"Encoder1_Base16")
            enc4 = tf.concat([enc4a,enc4b],axis =-1)
            
            ''' Decoder structure'''
            dec4 = self.DecoderBlockBase2(enc4,64,phase,"Decoder1_Base2")
            dec3 = self.DecoderBlockBase2(dec4 + enc3,32,phase,"Decoder2_Base2")
#             dec3b =  self.DecoderBase4(enc4,16,phase,"Decoder1_Base4")
    
#             dec3 = tf.concat([dec3a,dec3b],axis=-1)
            dec2 = self.DecoderBlockBase2(dec3 + enc2,8,phase,"Decoder3_Base2")
#             dec2b = self.DecoderBase8(dec3 + enc2,4,phase,"Decoder1_Base8")
#             dec2 = tf.concat([dec2a,dec2b],axis=-1)
            dec1 = self.DecoderBlockBase2(dec2 + enc1,8,phase,"Decoder4_Base2")
#             dec1b = self.DecoderBlockBase16(dec3,4,phase,"Decoder1_Base16")
#             dec1 = tf.concat([dec1a,dec1b],axis = -1)
            logits =tf.layers.conv2d(dec1,filters=3,padding="same",activation = None,kernel_size = (5,5),name= "preFinal")
            logits =tf.layers.conv2d(logits,filters=3,padding = "same",activation = None,kernel_size = (3,3),name= "logits")
            return logits
        
