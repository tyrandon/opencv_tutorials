import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture()
# initialize the Vision class
vision_shop = Vision()

'''
# https://www.crazygames.com/game/guns-and-bottle
wincap = WindowCapture()
vision_gunsnbottle = Vision('gunsnbottle.jpg')
'''

#loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

    # display the processed image
    #points = vision_limestone.find(screenshot, 0.5, 'rectangles')
    #points = vision_gunsnbottle.find(screenshot, 0.7, 'points')

    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    #loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    #if cv.waitKey(1) == ord('c'):
        #vision_shop.getCrop(screenshot)
    pressedKey = cv.waitKey(1)
    if pressedKey == ord('m'):
        #print('ay u pressed m')
        shop = vision_shop.findShopUnits(screenshot, 0.8)
        #print('the thing finished')
    elif pressedKey == ord('q'):
        cv.destroyAllWindows()
        break
    elif pressedKey == ord('1') or pressedKey == ord('2') or pressedKey == ord('3') or pressedKey == ord('4') or pressedKey == ord('5'):
        vision_shop.buyShopUnit(pressedKey)
    else:
        cv.imshow('Computer Vision', screenshot)

print('Done.')
