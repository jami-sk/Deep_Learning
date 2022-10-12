# Usage
# python 05-resizing.py -i <path to image>
# python 05-resizing.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png


import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path of the image", default="D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)
(h,w,c) = image.shape
print(f"Image loaded with size (height x width x channels) : {h} x {w} x {c}")

# Maintaining the aspect ratio
# change both hieght and with at same ratios
new_height = 100
ratio = new_height/image.shape[0]
new_dim = (int(image.shape[1]*ratio), new_height)

resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized The height to 150 pixels", resized)


# maintaining aspect ratio using imutils

resized = imutils.resize(image, height=100)
cv2.imshow("Resized using imutils", resized)

# CV2 interpolation methods
methods = [("cv2.INTER_AREA", cv2.INTER_AREA), 
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]

for name, method in methods:
    print(f"[INFO] {name}")
    resize = imutils.resize(image, width=image.shape[1]*3, inter=method)
    cv2.imshow(f"Method: {name}", resized)
    cv2.waitKey(0)

