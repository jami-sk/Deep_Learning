# USAGE
# python 21-image-histograms.py -i <path to the image>

import matplotlib.pyplot as plt
import cv2
import argparse
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Black-car.jpg")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0,256])

plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0,256])

norm_hist = hist/hist.sum()

plt.figure()
plt.title("Grayscale Histogram Normalized")
plt.xlabel("Bins")
plt.ylabel("% of pixels")
plt.plot(norm_hist)
plt.xlim([0,256])



########## Color Histogram
chans = cv2.split(image)
colors = ("b", "g", "r")

plt.figure()
plt.axis("off")
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")

for (chan, clr) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color=clr)
    plt.xlim([0,256])

#create a new figure and then plot 2D color histogram for the channels
fig=plt.figure()
plt.title("2D histograms on 2 channels")
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[0],chans[1]],[0,1],None,[32,32],[0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color histogram fro B and G")
plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1],chans[2]],[0,1],None,[32,32],[0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color histogram fro G and R")
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[2],chans[0]],[0,1],None,[32,32],[0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color histogram fro R and B")
plt.colorbar(p)

print(f"2D histogram shape: {hist.shape} with values {hist.flatten().shape[0]}")


# Lets build a 3D histograms with 8 bins in each direction
hist = cv2.calcHist([image],[0,1,2], None, [8,8,8], [0,256,0,256,0,256])
print(f"3D histogram shape: {hist.shape} with {hist.flatten().shape[0]} values")

plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))



def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ("b","g","r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, clr) in zip(chans, colors):
        hist = cv2.calcHist([chan],[0],mask,[256],[0,256])
        plt.plot(hist, color=clr)
        plt.xlim([0,256])

plot_histogram(image, "Histogram of Original Image")
cv2.imshow("Original Image", image)

# construct a mask for our image -  white at regions we want and black at regions we dont want
mask = np.zeros(image.shape[:2],dtype="uint8")
cv2.rectangle(mask, (115,128),(550,459),255,-1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Applying the mask", masked)


plot_histogram(image, "Histogram for Masked image", mask=mask)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

