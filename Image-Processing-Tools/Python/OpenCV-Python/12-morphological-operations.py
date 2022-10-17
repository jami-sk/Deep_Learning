# Usage
# python 12-morphological-operations.py -i <image path>


# This portion covers some advanced concepts like
# Eorsion, Dialation
# Opening, Closing
# Morphological Gradient
# Black hat, Top hat (also called white hat)

# Morphological operations are used on Binary images and some time gray images 
# used to cleanep the image (usally the source is from thresholding operations and edge detection methods)

import argparse
import re
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the Image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Facebook.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Loaded Image", image)

# Erosion ---------------------------
for i in range(0,5):
    eroded = cv2.erode(gray.copy(), None, iterations=i+1) #default  kernal 3x3
    cv2.imshow(f"Eroded {i+1} time", eroded)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Dialation --------------------------
cv2.imshow("Loaded Image", image)
for i in range(0,5):
    dialated = cv2.dilate(gray.copy(),None, iterations=i+1) #default  kernal 3x3
    cv2.imshow(f"Dialted for {i+1} times", dialated)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Opening ------------------------------
image = cv2.imread("D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Pyimagesearch-Noise.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Loaded Image", image)
kernelSizes = [(3,3),(5,5),(7,7)]

for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow(f"Opening: {kernelSize}", opening)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Closing ------------------------------
image = cv2.imread("D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Facebook.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Loaded Image", image)

for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow(f"Clsoing: {kernelSize}", closing)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphological Gradient
cv2.imshow("Loaded Image", image)
for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow(f"Gradient: {kernelSize}", gradient)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# Top Hat (Difference between original gray image and Opening)
# A top hat operation is used to reveal bright regions of an image on dark backgrounds.
image = cv2.imread("D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Black-car.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Loaded Image", image)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
topHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

cv2.imshow(f"Top Hat: {(13, 5)}", topHat)
cv2.waitKey(0)

# Balck Hat )Difference between gray image and Closing)
# A Black hat operation is used to reveal lighter regios against a dark brackground
blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

cv2.imshow(f"Black Hat: {(13, 5)}", blackHat)
cv2.waitKey(0)

cv2.destroyAllWindows()
