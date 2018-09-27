#!/usr/bin/env python
# coding: utf-8

# In[1]:


from RGBHallucinator import Hallucinator
import os
import time
import signal


# In[ ]:



pid_list =[]
pid = os.fork()
if pid == 0 :
    H_s2= Hallucinator('config.ini',2,0)
    H_s2.train()
else :
    pid_list.append(os.getpid())
    pid=os.fork()
    if pid == 0 :
        H_s4= Hallucinator('config.ini',4,1)
        H_s4.train()
    else :
        pid_list.append(os.getpid())
        pid = os.fork()
        if pid == 0 :
            H_s8= Hallucinator('config.ini',8,2)
            H_s8.train()
        else :
            pid_list.append(os.getpid())
            H_s16=Hallucinator('config.ini',16,2)
            H_s16.train()
            for p in pid_list:        
                os.waitpid(p,0)
            


print ("Training procedures of all scales completed")


# In[ ]:




