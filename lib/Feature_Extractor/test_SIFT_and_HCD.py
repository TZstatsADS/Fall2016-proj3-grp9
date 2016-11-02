# Author: YJ 
# Description: this python script tests SIFT and Harris Corner Detection in OpenCV (computer vision) on 
# chicken images from the Github repo whose path is Fall2016-proj3-grp9/data/images/. The output images 
# are saved to the github folder, whose path is Fall2016-proj3-grp9/figs/HCD_test_images/. 

# 0. Import pacakges  ##########################################################################################################
import numpy as np
import cv2

# 1. SIFT-dense ################################################################################################################
# read images  
img = cv2.imread('/Users/yanjin1993/GitHub/Fall2016-proj3-grp9/data/images/chicken_0002.jpg') # change to your local path 
# change background to gray 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# SIFT feature extractions 
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
# SIFT result image descriptions 
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# add keypoints to images (colorful)
img = cv2.drawKeypoints(gray,kps, img)
# save feature-extracted images to local 
cv2.imwrite('/Users/yanjin1993/Google Drive/Columbia University /2016 Fall /Applied Data Science /Project_003/image_classification/data /exported_data/HCD_images/chicken_002_SIFT.jpg',img)
descs_list = descs.tolist()


# 2. Harris Corner Detection ###################################################################################################
# read images  
img2 = cv2.imread('/Users/yanjin1993/GitHub/Fall2016-proj3-grp9/data/images/chicken_0002.jpg')
# change background to gray 
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
# HCD feature extractions 
dst = cv2.cornerHarris(gray2,2,3,0.04)
# result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img2[dst>0.01*dst.max()]=[0,0,255]
# save feature-extracted images to local 
cv2.imwrite('/Users/yanjin1993/Google Drive/Columbia University /2016 Fall /Applied Data Science /Project_003/image_classification/data /exported_data/HCD_images/chicken_002_HDC.jpg',img2)
