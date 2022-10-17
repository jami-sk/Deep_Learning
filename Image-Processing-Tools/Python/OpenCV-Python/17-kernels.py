# Usage
# python 17-kernels.py -i <image path>

import argparse
import cv2 
from skimage.exposure import rescale_intensity
import numpy as np

def convolve(image, kernel):
    (iH, iW)  = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW-1)//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH,iW), dtype="float32")

    for y in np.arange(pad,iH+pad):
        for x in np.arange(pad, iW+pad):
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
            k = (roi*kernel).sum()
            output[y-pad,x-pad] = k
    
    output = rescale_intensity(output, in_range=(0,255))
    output = (output*255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Construct average blurring kernels used to smooth an image
smallBlur = np.ones((7,7), dtype="float")*(1.0/(7*7))
largeBlur = np.ones((21,21), dtype="float")*(1.0/(21*21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, -1]), dtype="int")

# construct a laplacian kernel used to detect edge-like regions of an image
laplacain = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# Construct soble x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct Soble y-axis kernel
sobleY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 2]), dtype="int")

# construct a kernel bank
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacain),
    ("sobel_x", sobelX),
    ("sobel_y", sobleY))

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, kernel) in kernelBank:
    print(f"[INFO] applying {kernelName} kernel")
    convolveOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    cv2.imshow("Original", image)
    cv2.imshow(f"{kernelName} - convolve", convolveOutput)
    cv2.imshow(f"{kernelName} - opencv", opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





