# Usage
# python 08-image-arithmatics.py -i <path of the image>
# python 08-image-arithmatics.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png

import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help = "Path of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loade Image", image)


# Cv2 clips the values to [0 255] limits
added = cv2.add(np.uint8([200]), np.uint8([100]))
subtracted = cv2.subtract(np.uint8([50]), np.uint8([100]))
print(f"CV2 addition result: {added}")
print(f"CV2 Subtraction result: {subtracted}")

# but numpy loops over the values after [0 255]
added = np.uint8([200]) + np.uint8([100])
subtracted = np.uint8([50]) - np.uint8([100])
print(f"Numpy addition result: {added}")
print(f"Numpy Subtraction result: {subtracted}")

# Brightening theimage using addition
M = np.ones(image.shape, dtype="uint8") *100
bright = cv2.add(image, M)
cv2.imshow("Birghtened Image", bright)

# Darkening the image using subratcion
M = np.ones(image.shape, dtype="uint8")*50
dark = cv2.subtract(image, M)
cv2.imshow("Darker Version", dark)
cv2.waitKey(0)


