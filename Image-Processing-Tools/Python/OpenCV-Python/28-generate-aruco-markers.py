# USAGE
# python 28-generate-aruco-markers.py --id 24 --type DICT_5X5_100 --output tags/DICT_5X%_100_id24.png

import numpy as np
import argparse 
import cv2, sys 

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help = "path to output containing ArUco tag")
ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUco tag to generate")
ap.add_argument("-t", "--type", type=str, default = "DICT_ARUCO_ORIGINAL", help=  "type of ArUco tag to generate")
args = vars(ap.parse_args())

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL":cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5":cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9":cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10":cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11":cv2.aruco.DICT_APRILTAG_36h11,
    
}

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"[INFO] genrating ArUco tag of {args['type']} is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

print(f"[INFO] Generating ArUco tag type {args['type']} with ID {args['id']}")
tag = np.zeros((300,300,1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, 1) # final value is padding arond the tag

cv2.imwrite(args["output"], tag)
cv2.imshow("ArUco Tag", tag)
