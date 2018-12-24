import os
import pandas as pd
import numpy as np

# np.set_printoptions(threshold=np.inf)

import cv2
import openslide
import scipy.misc
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

# read contours coordinates
# save them as an array
data_path=os.getcwd()+'/contours.txt'
data=pd.read_csv(data_path, sep='\s+',header=None, names=['vertex','X', 'Y'])
cols = data.shape[1]
X_c=data.iloc[:,1:cols-1]
y_c = data.iloc[:,cols-1:cols]
X_np = np.array(X_c.values)
Y_np=np.array(y_c.values)
X=[]
Y=[]

for i in range(X_np.shape[0]):
    X.append(filter(str.isdigit,X_np[i,0]))
    Y.append(filter(str.isdigit,Y_np[i,0]))

X=np.array(X,dtype=np.float32)
Y=np.array(Y,dtype=np.float32)


X_min=np.int(np.min(X))
Y_min=np.int(np.min(Y))
X_max=np.int(np.max(X))
Y_max=np.int(np.max(Y))

X=X-X_min
X=X.astype(int)
Y=Y-Y_min
Y=Y.astype(int)

X=X.reshape(1,-1)
Y=Y.reshape(1,-1)
# vector=np.vstack((Y,X))

# read tiff images using opencv
img_path = os.getcwd() + '/OU11-006-01-01_003_CIN 2.tif'

patch=cv2.imread(img_path)
gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(gray,127,255,0)
# f_cnt, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# draw contours on patch
contour=[]

cnt_arr=np.zeros((X.shape[1],1,2),dtype=np.int32)
for i in range(X.shape[1]):
	cnt_arr[i][0][0]=X[0][i]
	cnt_arr[i][0][1]=Y[0][i]

contour.append(cnt_arr)

# print(len(contour))

cnt=contour[0]

# print(cnt.shape)

# create a mask
h, w = patch.shape[:2]
mask = np.zeros(patch.shape, np.uint8)
cv2.drawContours(mask,contour,0,(1,1,1),-1)

im_cnt=patch*mask

# extract patch within the given contour
# for i in range(h):
# 	for j in range(w):
# 		if (im_cnt[i,j,:]==0).any():
# 			im_cnt[i,j,:]=255
im_cnt2=im_cnt.copy()
im_cnt2[im_cnt==0]=255
# further extract ROI

im_cnt_gray = cv2.cvtColor(im_cnt2,cv2.COLOR_BGR2GRAY)
# method1: using cv2.threshold()
# as the histogram shows, the grayscale has two peaks

# plt.figure()
# plt.subplot(221),plt.imshow(gray,'gray')
# plt.subplot(222),plt.hist(gray.ravel(),256)
# plt.subplot(223),plt.imshow(im_th,'gray')
# plb.show()
ret, im_th = cv2.threshold(im_cnt_gray,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
f_cnt, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# method2: using adaptive threshold
# im_th = cv2.adaptiveThreshold(im_cnt_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
# f_cnt, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

idx_list=[len(one) for one in f_cnt]

idx=idx_list.index(max(idx_list))

cv2.drawContours(im_cnt,[f_cnt[idx]],0,(0,255,0),1)

# create a mask again to extract the ROI
mask2 = np.zeros(patch.shape, np.uint8)
cv2.drawContours(mask2,[f_cnt[idx]],0,(1,1,1),-1)
im_cnt_again=im_cnt2*mask2
im_cnt_again2=im_cnt_again.copy()
# for i in range(h):
# 	for j in range(w):
# 		if (im_cnt2[i,j,:]==0).any():
# 			im_cnt2[i,j,:]=255
im_cnt_again2[im_cnt_again==0]=255
# cv2.imshow("mask",mask)
# cv2.imshow("Thresholded Image", im_th)
# cv2.imshow("im_cnt",im_cnt)
cv2.imshow("im_ext",im_cnt_again2)
cv2.waitKey(0)
cv2.destroyAllWindows()




# save extracted image to the disk
# scipy.misc.imsave(os.getcwd()+'/OU11-006-test.tif',tile_after)