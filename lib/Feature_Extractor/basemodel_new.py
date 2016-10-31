import cv2
import os
import numpy as np
import pandas as pd

name=pd.read_csv('/Users/youzhuliu/Desktop/哥大/fall 2016/5243 ads/project3/Fall2016-proj3-grp9/dat_namelist.csv')
X_train_name=name['img_directory']

i=0
image=cv2.imread(X_train_name[0], 0)

#####MSER
img_m = cv2.resize(image,(256,256),interpolation = cv2.INTER_LINEAR)
vis = img_m.copy()
mser = cv2.MSER_create()
regions = mser.detectRegions(img_m, None)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
ms=cv2.polylines(vis, hulls, 1, (0, 255, 0))
a = np.reshape(ms, (1,np.product(ms.shape)))[0]
b=a.tolist()
df_ms=pd.DataFrame(b,columns=[X_train_name[0]])

i=i+1

for i in range(len(X_train_name)):
	image = cv2.imread(X_train_name[i], 0)
	img_m = cv2.resize(image,(256,256),interpolation = cv2.INTER_LINEAR)
	vis = img_m.copy()
    mser = cv2.MSER_create()
    regions = mser.detectRegions(img_m, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    ms=cv2.polylines(vis, hulls, 1, (0, 255, 0))
    a = np.reshape(ms, (1,np.product(ms.shape)))[0]
    b=a.tolist()
    df_ms_i=pd.DataFrame(b,columns=[X_train_name[i]])
    df_ms=pd.concat([df_ms,df_ms_i],axis=1)
    i +=1
    print i

##### df_ms2=df_ms.drop(df_ms.columns[0],1,inplace=True)

df_ms.to_csv('df_ms.csv',index=False)


####HOG Methods
j=0
img_h=cv2.resize(image,(128,128),interpolation = cv2.INTER_LINEAR)
hog = cv2.HOGDescriptor()
h = hog.compute(img_h)
c= np.reshape(h, (1,np.product(h.shape)))[0]
d=c.tolist()
df_h=pd.DataFrame(d,columns=[X_train_name[0]])

j=j+1
for j in range(len(X_train_name)):
	image = cv2.imread(X_train_name[j], 0)
	img_h=cv2.resize(image,(128,128),interpolation = cv2.INTER_LINEAR)
	hog = cv2.HOGDescriptor()
	h = hog.compute(img_h)
    c= np.reshape(h, (1,np.product(h.shape)))[0]
    d=c.tolist()
    df_h_j=pd.DataFrame(d,columns=[X_train_name[j]])
    df_h=pd.concat([df_h,df_h_j],axis=1)
    j +=1
    print j

df_h.to_csv('df_h.csv',index=False)

####HARRIS CORNER DETECTION
k=0
img_hc=cv2.resize(image,(256,256),interpolation = cv2.INTER_LINEAR)
gray = np.float32(img_hc)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
e = np.reshape(dst, (1,np.product(dst.shape)))[0]
f=e.tolist()
df_hc=pd.DataFrame(f,columns=[X_train_name[0]])

k=k+1
for k in range(len(X_train_name)):
	image = cv2.imread(X_train_name[k], 0)
	img_hc=cv2.resize(image,(256,256),interpolation = cv2.INTER_LINEAR)
	gray = np.float32(img_hc)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    e = np.reshape(dst, (1,np.product(dst.shape)))[0]
    f=e.tolist()
    df_hc_k=pd.DataFrame(f,columns=[X_train_name[k]])
    df_hc=pd.concat([df_hc,df_hc_k],axis=1)
    k +=1
    print k

df_hc.to_csv('df_hc.csv',index=False)
