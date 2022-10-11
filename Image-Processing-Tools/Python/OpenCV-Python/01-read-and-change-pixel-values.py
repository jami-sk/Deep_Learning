# Usage
# python 01-read-and-change-pixel-values.py -i <image path>
# python 01-read-and-change-pixel-values.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Range-Rover-Evoque.jpg


# importing packages
import argparse
import cv2

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--image", help="Path of the Image")
args = vars(arg_parse.parse_args())


image = cv2.imread(args["image"])
(h,w,c) = image.shape

cv2.imshow("Loaded Image", image)
print(f"Image has loaded and of size (height x width x channels) : {h} x {w} x {c}")

# Read image pixels at different Locations
(b, g, r) = image[0,0]
print(f"Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}")

(b, g, r) = image[h//2, w//2]
print(f"Pixel at ({h//2}, {w//2}) - Red: {r}, Green: {g}, Blue: {b}")


# Set pixel value at different location

image[h//2, w//2] = (0,255,0)
(b, g, r) = image[h//2, w//2]
print(f"Pixel at ({h//2}, {w//2}) - Red: {r}, Green: {g}, Blue: {b}")


# Slicing the image
(cX, cY) = (w//2, h//2)

tl = image[:cY, :cX]
cv2.imshow("Top Left Image", tl)
tr = image[:cY, cX:]
cv2.imshow("Top Right Image", tr)
br = image[cY:, cX:]
cv2.imshow("Bottom Right Image", br)
bl = image[cY:, :cX]
cv2.imshow("Bottom Left Image", bl)

# Setting up values at different level
image[:cY, :cX] = (0,255,0)
image[:cY, cX:] = (255,0,0)
image[cY:, cX:] = (0,0,255)
image[cY:, :cX] = (0,0,0)

cv2.imshow("Modified Image", image)
cv2.waitKey(0)





