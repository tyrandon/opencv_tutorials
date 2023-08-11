import cv2 as cv
import numpy as np
import os
from enum import Enum


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tftChampionPoolSizes = [(0, 0), (1, 14), (14, 27), (27, 40), (40, 52), (52, 60), (60, 64)]
tftUnitPositions = [(850, 880), (1115, 1145), (1385, 1415), (1655, 1685), (1925, 1955)]

def findClickPositions(needle_img_path, haystack_img_path, threshold=0.8, debug_mode=None):
        
    # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
    # Save the dimensions of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # There are 6 methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(haystack_img, needle_img, method)

    # Get the all the positions from the match result that exceed our threshold
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    #print(locations)

    # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
    # locations by using groupRectangles().
    # First we need to create the list of [x, y, w, h] rectangles
    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        # Add every box to the list twice in order to retain single (non-overlapping) boxes
        rectangles.append(rect)
        rectangles.append(rect)
    # Apply group rectangles.
    # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
    # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
    # in the result. I've set eps to 0.5, which is:
    # "Relative difference between sides of the rectangles to merge them into a group."
    rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
    #print(rectangles)

    points = []
    if len(rectangles):
        #print('Found needle.')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        # Loop over all the rectangles
        for (x, y, w, h) in rectangles:

            # Determine the center position
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            # Save the points
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                # Determine the box position
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                # Draw the box
                cv.rectangle(haystack_img, top_left, bottom_right, color=line_color, 
                             lineType=line_type, thickness=2)
            elif debug_mode == 'points':
                # Draw the center point
                cv.drawMarker(haystack_img, (center_x, center_y), 
                              color=marker_color, markerType=marker_type, 
                              markerSize=40, thickness=2)

        if debug_mode:
            cv.imshow('Matches', haystack_img)
            cv.waitKey()
            #cv.imwrite('result_click_point.jpg', haystack_img)

    return points

def identifyUnit(haystack_img_path, position=(0, 0), counter=0):
    temp_cropped_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)[position[1] - 172 : position[1] - 3, position[0] - 230 : position[0] + 40]
    cv.imwrite('temp_cropped_img.png', temp_cropped_img)
    points = []
    poolSize = tftChampionPoolSizes[counter]
    for num in range(poolSize[0], poolSize[1]):
        points = findClickPositions('Images/ChampionImages/TFT_Champion_Img' + str(num) + '.png', 'temp_cropped_img.png', 0.8, None)
        if (len(points) > 0):
            return TFTChampions(num).name
    print('Champion not found.')
    return ''

def findShopUnits(haystack_img_path, threshold=0.8, debug_mode=None):
    unitPositions = []
    for num in range(1, 7):
        points = findClickPositions('Images/OtherTFTImages/TFT_Cost' + str(num) + '.png', haystack_img_path, threshold, debug_mode)
        unitPositions.append(points)
        #print(points)
    #print(unitPositions)
    counter = 0
    realUnitPositions = []
    for cost in unitPositions:
        counter+=1
        for position in cost:
            if position[0] > 500:
                championName = identifyUnit(haystack_img_path, position, counter)
                realUnitPositions.append((position, championName))
    realUnitPositions = sorted(realUnitPositions)
    #print(realUnitPositions)
    shop = []
    counter = 0
    for pos in range(5):
        if tftUnitPositions[pos][0] < realUnitPositions[counter][0][0] < tftUnitPositions[pos][1]:
            shop.append(realUnitPositions[counter][1])
            counter+=1
        else:
            shop.append(TFTChampions(0).name)
    print(shop)
    print('Done')

class TFTChampions(Enum):
    NOCHAMPION = 0
    CASSIOPEIA = 1
    CHOGATH = 2
    IRELIA = 3
    JHIN = 4
    KAYLE = 5
    MALZAHAR = 6
    MAOKAI = 7
    ORIANNA = 8
    POPPY = 9
    RENEKTON = 10
    SAMIRA = 11
    TRISTANA = 12
    VIEGO = 13
    ASHE = 14
    GALIO = 15
    JINX = 16
    KASSADIN = 17
    KLED = 18
    SETT = 19
    SORAKA = 20
    SWAIN = 21
    TALIYAH = 22
    TEEMO = 23
    VI = 24
    WARWICK = 25
    ZED = 26
    AKSHAN = 27
    DARIUS = 28
    EKKO = 29
    GAREN = 30
    JAYCE = 31
    KALISTA = 32
    KARMA = 33
    KATARINA = 34
    LISSANRA = 35
    REKSAI = 36
    SONA = 37
    TARIC = 38
    VELKOZ = 39
    APHELIOS = 40
    AZIR = 41
    GWEN = 42
    JARVAN = 43
    KAISA = 44
    LUX = 45
    NASUS = 46
    SEJUANI = 47
    SHEN = 48
    URGOT = 49
    YASUO = 50
    ZERI = 51
    AATROX = 52
    AHRI = 53
    BELVETH = 54
    HEIMERDINGER = 55
    KSANTE = 56
    RYZE = 57
    SENNA = 58
    SION = 59
    GOLDINATOR = 60
    MECHANOSWARM = 61
    SELFREPAIR = 62
    SHRINKMODULE = 63


#points = findClickPositions('Images/ChampionImages/TFT_Champion_Img41.png', 'Images/TFTScreenshots/TFT_Screenshot2.png', threshold=0.70, debug_mode='rectangles')
#print(points)
#print('Done.')
findShopUnits('Images/TFTScreenshots/TFT_Screenshot1.png', threshold=0.8, debug_mode=None)
findShopUnits('Images/TFTScreenshots/TFT_Screenshot2.png', threshold=0.8, debug_mode=None)
