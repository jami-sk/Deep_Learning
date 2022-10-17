# Usage
# python 14-color-spaces.py -i <path of the image>


import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Black-car.jpg")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image into HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Format", hsv)

for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image into l*a*b space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("l*a*b", lab)

for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(hsv)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert BGR to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Loaded Image", image)
cv2.imshow("Gray image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()