import sys
import numpy as np
import cv2 as cv

im = cv.imread('Images/NumberRecognition/pitrain.png', cv.IMREAD_UNCHANGED)
im3 = im.copy()

gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),0)
thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv.contourArea(cnt)>50:
        [x,y,w,h] = cv.boundingRect(cnt)
        
        if  h>28:
            cv.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv.resize(roi,(10,10))
            cv.imshow('norm',im)
            key = cv.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)