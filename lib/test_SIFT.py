# Author: YJ 
# Description: this python script tests SIFT and Harris Corner Detection in OpenCV (computer vision) on 
# chicken images from the Github repo whose path is Fall2016-proj3-grp9/data/images/. The output images 
# are saved to 

# test_sift.py
import numpy as np
import cv2
img = cv2.imread('/Users/yanjin1993/GitHub/Fall2016-proj3-grp9/data/images/chicken_0002.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
img=cv2.drawKeypoints(gray,kps, img)
cv2.imwrite('/Users/yanjin1993/Google Drive/Columbia University /2016 Fall /Applied Data Science /Project_003/image_classification/data /exported_data/HCD_images/chicken_002_SIFT.jpg',img)
descs_list = descs.tolist()
