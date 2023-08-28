import cv2 as cv
import numpy as np
import os
from time import time

#######   training part    ###############
samples = np.loadtxt("tftsamples.data", np.float32)
responses = np.loadtxt("tftresponses.data", np.float32)
responses = responses.reshape((responses.size, 1))

model = cv.ml.KNearest_create()
model.train(samples, cv.ml.ROW_SAMPLE, responses)

############################# testing part  #########################
total_time = 0
path = "Images/NumberRecognition/GoldScreenshots/"
for filename in os.listdir(path):
    loop_time = time()
    im = cv.imread(str(path) + str(filename), cv.IMREAD_UNCHANGED)[1180:1210, 1152:1250]
    # out = np.zeros(im.shape,np.uint8)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    nums = []
    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            [x, y, w, h] = cv.boundingRect(cnt)
            if x < 50 and y < 20 and 10 < w < 20 and 20 < h < 30:
                cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y : y + h, x : x + w]
                roismall = cv.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                string = str(int((results[0][0])))
                nums.append((x, string))
    nums.sort()
    cost = ""
    for digit in nums:
        cost += str(digit[1])
    timeTaken = time() - loop_time
    print(cost)
    cv.imshow('im', im)
    cv.waitKey()
    print(timeTaken)
    total_time += timeTaken

# cv.imshow('im',im)
# cv.imshow('out',out)
#print(blahString[::-1])
print('Time Taken: ' + str(total_time))
# cv.waitKey(0)
