# TEST ON!!!!!! IF THE TA Laptop does not contain the data under caffe_root then 
# WARNING!!!!!!!!!!!

# STEP 0 Basic Setups ##########################################################################################
# 0.1 Initiate libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os # for local path 
import caffe  # for feature extraction 
from glob import glob 
from os import path
import sys



caffe_root='/Users/yanjin1993/caffe/'
sys.path.insert(0, caffe_root + 'python')

# An instance of RcParams for handling default matplotlib values ??????????
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# STEP 1 Download Reference Model (CaffeNet) ######################################################################

# Test existence of reference model in your local path 
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Need to download pre-trained CaffeNet model...'

# If not in terminal 
# /Users/yanjin1993/caffe/scripts/download_model_binary.py /Users/yanjin1993/caffe/models/bvlc_reference_caffenet


# STEP 2 Load Net and Set Up Input Pre-processing #####################################################################
# 2.1 Set Caffe to CPU mode and load the net from disk 
caffe.set_mode_cpu()

net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST) 


# 2.2 Set up input processing 
# Customizable code here, we use caffe.io.Transformer here 
###### ??????? Why do we need transformation here?
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# 2.3 Create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 2.4 Set the size of the input 
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


# STEP 3. Build Images' Directory File ################################################################################
# 3.1 Generate images' directories list  
#def get_files_in(folder, pattern='*.txt'):
 #   return glob(path.join(folder, pattern))

# 3.2 File ???????????????
#def filenames(folder):
#    filename = get_files_in(folder, '*.jpg')
#    #filename1 = get_files_in(folder, '*.mat')
#    #for i in range(len(filename)):
#        #filename.append(filename1[i])
#    return filename
