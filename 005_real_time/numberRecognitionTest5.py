import cv2 as cv
import pytesseract
from time import time

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

loop_time = time()
# Grayscale, Gaussian blur, Otsu's threshold
image = cv.imread('Images/NumberRecognition/pi.png')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(gray,255,1,1,11,2)

# Perform text extraction
data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
print(data)
timeTaken = time() - loop_time
print(timeTaken)

#cv.imshow('thresh', thresh)
#cv.waitKey()