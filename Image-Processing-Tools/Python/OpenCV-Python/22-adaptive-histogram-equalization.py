# USAGE
# python 22-adaptive-histogram-equalization.py -i <path to the image>

import argparse
from operator import eq
import cv2 
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to the Image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Black-car.jpg")
ap.add_argument("-c", "--clip", type=float, default = 2.0, help="threshold for contrast limiting")
ap.add_argument("-t", "--tile", type=int, default=8, help="tile grid size --  divides image into tile x tile cells")
args = vars(ap.parse_args())

print(f"[INFO] Loading input image")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFI] performing histogram equalization")
# Global histogram equalization
equalized = cv2.equalizeHist(gray)

# Show the original gray scale image and equalized image
cv2.imshow("Input Gary", gray)
cv2.imshow("Histogram Equalization", equalized)


print("[INFO] applying CLAHE")
clahe = cv2.createCLAHE(clipLimit=args["clip"], tileGridSize=(args["tile"],args["tile"]))
equalized = clahe.apply(gray)

cv2.imshow("CLAHE",equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()