# Usage
# python 16-adaptive-thresholding.py -i <image path>

import argparse
from ast import arg
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the Image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\steve-jobs-business-card.jpg")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 0)

# Apply simple thresholding with a hardcoded threshold value
(T, thresholdInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Simple Thresholding", thresholdInv)
cv2.waitKey(0)

# Apply OTSU's automatic thresholding
(T, threshOTSU) = cv2.threshold(blurred, 0,255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cv2.imshow("OTSU Thresholding", threshOTSU)
cv2.waitKey(0)

# Adaptive thresholding is an local thresholding rather than global thresholding
# Main assumption is smaller regions of an image tends to have more uniform illumination
# T = mean(local region) - C 

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean Adaptive Thresholding", thresh)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
cv2.imshow("Gaussian Adaptive Thresholding", thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()
