import cv2 as cv
import numpy as np
from time import time

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv.ml.KNearest_create()
model.train(samples, cv.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

loop_time = time()
im = cv.imread('Images/NumberRecognition/pi.png')
#out = np.zeros(im.shape,np.uint8)
gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(gray,255,1,1,11,2)

contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
outputString = []
blahString = ""
for cnt in contours:
    if cv.contourArea(cnt)>50:
        [x,y,w,h] = cv.boundingRect(cnt)
        if h>28:
            cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            outputString.append((x, string))
            if len(outputString) == 25:                
                outputString.sort()
                tempString = ""
                for string in outputString:
                    tempString += string[1]
                tempString = tempString[::-1]
                blahString += tempString
                outputString.clear()
            #cv.putText(out,string,(x,y+h),0,1,(0,255,0))

#cv.imshow('im',im)
#cv.imshow('out',out)
print(blahString[::-1])
print(time() - loop_time)
#cv.waitKey(0)