# Usage
# python 18-image-gradients.py -i <path to the image> -s <schar(1) kernel or sobel(0)>

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path to the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\coins.jpg")
ap.add_argument("-s", "--scharr", type=int, default=0, help="Apply Schar Kernel or not")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)

ksize = -1 if args["scharr"] >0 else 3
gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

# Gradient is an floating point operation. To convert it into unsigned integer for ither applications we need to do follwoing step
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Compute gradients along X and Y directions respectively
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

magnitude = np.sqrt(gX**2 + gY**2)
orientation = np.arctan2(gY,gX)*(180/np.pi)%100

(fig,axs) = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
axs[0].imshow(gray, cmap="gray")
axs[1].imshow(magnitude, cmap="jet")
axs[2].imshow(orientation, cmap="jet")

for i in range(0,3):
    axs[i].get_xaxis().set_ticks([])
    axs[i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.show()
