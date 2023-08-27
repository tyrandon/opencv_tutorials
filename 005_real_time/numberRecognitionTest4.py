import cv2 as cv
import numpy as np
from time import time
from vision import Vision

vision_thing = Vision()
blahString = ''
loop_time = time()
im = cv.imread('Images/NumberRecognition/pi.png')
#out = np.zeros(im.shape,np.uint8)
gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(gray,255,1,1,11,2)
contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
outputString = []
for cnt in contours:
    if cv.contourArea(cnt)>50:
        [x,y,w,h] = cv.boundingRect(cnt)
        if h > 28:
            found = False
            for i in range(10):
                temp_im = im[y-2 : y+h+2, x : x+h]
                temp_im = cv.cvtColor(temp_im, cv.COLOR_BGR2GRAY)
                needle_img = cv.imread('Images/NumberRecognition/pi' + str(i) + '.png')
                needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)
                points = vision_thing.findClickPositions(needle_img, temp_im, 0.7, None)     
                if len(points) > 0:
                    outputString.append((x, i))
                    if len(outputString) == 25:
                        tempString = ''
                        outputString.sort()
                        for num in outputString:
                            tempString += str(num[1])
                        tempString = tempString[::-1]
                        blahString += tempString
                        outputString.clear()
                    found=True
                    break
            if not found:
                cv.imshow('blah', im[y-2 : y+h+2, x : x+h])
                cv.waitKey()
                    
#cv.imshow('im',im)
#cv.imshow('out',out)
print(blahString[::-1])
timeTaken = time() - loop_time
print(timeTaken)
#cv.waitKey(0)