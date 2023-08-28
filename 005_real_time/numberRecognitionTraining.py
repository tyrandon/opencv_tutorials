import sys
import numpy as np
import cv2 as cv
import os

path = "Images/NumberRecognition/GoldScreenshots/"
for filename in os.listdir(path):
    im = cv.imread(str(path) + str(filename), cv.IMREAD_UNCHANGED)[1180:1210, 1152:1250]
    im3 = im.copy()

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    #################      Now finding Contours         ###################

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0, 100))
    responses = []
    keys = [i for i in range(48, 58)]

    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            [x, y, w, h] = cv.boundingRect(cnt)

            if x < 50 and y < 20 and 10 < w < 20 and 20 < h < 30:
                cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[y : y + h, x : x + w]
                roismall = cv.resize(roi, (10, 10))
                winname = "Test"
                cv.namedWindow(winname)  # Create a named window
                cv.moveWindow(winname, 40, 30)  # Move it to (40,30)
                cv.imshow(winname, im)
                # cv.imshow('norm',im)
                key = cv.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt("tftsamples.data", samples)
np.savetxt("tftresponses.data", responses)
