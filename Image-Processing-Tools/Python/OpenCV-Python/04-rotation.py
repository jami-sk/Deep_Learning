# Usage
# python 04-rotation.py -i <image path>
# python 04-rotation.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png


import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-1", "--image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png", help="Path of the Image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)
(h,w) = image.shape[:2]
(cX, cY) = (w//2, h//2)

# Rotate the image by 45 degrees around the center of the image
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by 45", rotated)

# Rotate image by -90 degress around the center
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by -90", rotated)

# Roatate image at random arbitrary point rather than center
M = cv2.getRotationMatrix2D((20,20), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by 45 at arbitrary point", rotated)


# Rotate image using imutils
rotated = imutils.rotate(image, 45)
cv2.imshow("Rotated by 45", rotated)

# Rotataion of image without cropping
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Rotated by 45 with out cropping", rotated)
cv2.waitKey(0)






