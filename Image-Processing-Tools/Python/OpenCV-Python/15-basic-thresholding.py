# Usage
# python 15-basic-thresholding.py -i <path of the image>

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
blurred = cv2.GaussianBlur(gray, (7,7), 0)
cv2.imshow("Gaussian Blurred Image", blurred)


# Basic Thresholding 
# check each pixel value greater than threshold value (200) and assign black, otherwsie assign white
(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

# Normal Thresholding
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

# Visualize only masked regions in the image
masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("Output", masked)
cv2.waitKey(0)

#OTSU hresholding (Assumes binomial distribution)
(T, threshInvOtsu) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cv2.imshow("Threshold Binary Inverse or OTSU", threshInvOtsu)
print(f"[INFO] otsu's thresholding value : {T}")

# Visualize only masked regions in the image
masked = cv2.bitwise_and(image, image, mask=threshInvOtsu)
cv2.imshow("Output", masked)
cv2.waitKey(0)