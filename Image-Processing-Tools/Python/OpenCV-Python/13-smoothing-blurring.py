# Usage
# python 13-smoothing-blurring.py -i <path of image>


# In this portion we are going to learn blurring using
# Simple Blurring (cv2.blur), Weighted Gaussian Blurring (cv2.gaussianblur)
# Median Blurring (cv2.medianblur), Bilateral blurring (cv2.bilateralFilter)

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Black-car.jpg")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

kernelSizes = [(3,3), (9,9), (15,15)]

# Simple Blurring (cv2.blur)
for kernelSize in kernelSizes:
    blurred = cv2.blur(image, kernelSize)
    cv2.imshow(f"Average Blurring: {kernelSize} ", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# Weighted Gaussian Burring (cv2.gaussianBlur)
cv2.imshow("Loaded Image", image)
for kernelSize in kernelSizes:
    blurred = cv2.GaussianBlur(image, kernelSize, 0) # 0 for Opencv automaticaaly compute sigma (standard deviation)
    cv2.imshow(f"Gaussian Blurring: {kernelSize} ", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Median Blurring (cv2.medianblur)
cv2.imshow("Loaded Image", image)
for kernelSize in kernelSizes:
    blurred = cv2.medianBlur(image, kernelSize[0]) # Opencv allows only square kernels for Median Blur
    cv2.imshow(f"Median Blurring: {kernelSize} ", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# Bilateral Blurring (cv2.bilateralFilter)
cv2.imshow("Loaded Image", image)
params = [(11,21,7), (11,41,21), (11,61,39)]

for (diameter, sigmaColor, sigmaSpace) in params:
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    cv2.imshow(f"Biltaeral Blurred: d={diameter}, sc={sigmaColor}, ss={sigmaSpace}", blurred)
    cv2.waitKey(0)
cv2.destroyAllWindows()