# USAGE
# python 25-automatic-color-correction.py -i <path to reference image> --input <path to test image>

from symbol import parameters
from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import cv2
import imutils
import sys

def find_color_card(image):
    aurcoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, aurcoDict, parameters=arucoParams)

    try:
        ids = ids.flatten()
        # Extract Top Left Corner
        i = np.squeeze(np.where(ids==923))
        topLeft = np.squeeze(corners[i])[0]
        # Ectract top right corner
        i = np.squeeze(np.where(ids==1001))
        topRight = np.squeeze(corners[i])[0]
        # Ectract bottom right corner
        i = np.squeeze(np.where(ids==241))
        bottomRight = np.squeeze(corners[i])[0]
        # Ectract bottom left corner
        i = np.squeeze(np.where(ids==1007))
        bottomLeft = np.squeeze(corners[i])[0]
    except:
        return None
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)

    return card


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True, help="path t the input reference image")
ap.add_argument("-i", "--input", required=True, help = "path to the input image to apply color corection to")
args = vars(ap.parse_args())

print("[INFO] Loading Images...")
ref = cv2.imread(args["reference"])
image = cv2.imread(args["input"])

ref = imutils.resize(ref, width=600)
image = imutils.resize(image, width=600)

cv2.imshow("Reference", ref)
cv2.imshow("Input", image)

print("[INFO] finding colr matching cards...")
refCard = find_color_card(ref)
imageCard = find_color_card(image)

if refCard is None or imageCard is None:
    print("[INFO] could not find color matching card in both images")
    sys.exit(0)

cv2.imshow("Reference Color Card", refCard)
cv2.imshow("Input Color Card", imageCard)

print("[INFO] matching images...")
imageCard = exposure.match_histograms(imageCard, refCard, multichannel=True)

cv2.imshow("Input Color Card After Matching", imageCard)
cv2.waitKey(0)

