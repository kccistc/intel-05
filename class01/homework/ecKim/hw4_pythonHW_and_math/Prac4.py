import numpy as np
import matplotlib.pylab as plt
import cv2

img = cv2.imread('/home/eckim/workspace/math/smile3.jpg')
img = img.transpose()
img1=np.expand_dims(img,axis=0)
print(img.shape)
print(img1.shape)
# cv2.imshow('fig1',img)
# cv2.waitKey(0)
