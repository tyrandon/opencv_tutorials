import cv2 as cv
import numpy as np
from time import time
from vision import Vision

vision_thing = Vision()
counter = 1
totalTime = 0
while counter < 10:
    loop_time = time()
    im = cv.imread('Images/NumberRecognition/piReduced' + str(counter) + '.png')
    #out = np.zeros(im.shape,np.uint8)
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    outputString = []
    for cnt in contours:
        if cv.contourArea(cnt)>50:
            [x,y,w,h] = cv.boundingRect(cnt)
            if h > 28:
                for i in range(1, 10):
                    temp_im = im[y-2 : y+h+2, x : x+h-8]
                    temp_im = cv.cvtColor(temp_im, cv.COLOR_BGR2GRAY)
                    needle_img = cv.imread('Images/NumberRecognition/pi' + str(i) + '.png')
                    needle_img = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)
                    points = vision_thing.findClickPositions(needle_img, temp_im, 0.9, None)     
                    if len(points) > 0:
                        outputString.append((x, i))
                        break

    #cv.imshow('im',im)
    #cv.imshow('out',out)
    outputString.sort()
    blahString = ''
    for num in outputString:
        blahString += str(num[1])
    print(blahString)
    timeTaken = time() - loop_time
    print(timeTaken)
    totalTime += timeTaken
    counter+=1
    #cv.waitKey(0)
print(totalTime)