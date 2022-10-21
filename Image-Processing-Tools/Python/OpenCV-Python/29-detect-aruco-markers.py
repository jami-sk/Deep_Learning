# USAGE
# python 29-detect-aruco-markers.py -i <path to image> --t <DICT_5X5_100> -v <video or image>

import argparse, cv2 ,sys, imutils, time
from symbol import parameters
from xml.dom.expatbuilder import Rejecter
from imutils.video import videostream

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=0, help="path to input image containing ArUco tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUco tag to detect")
ap.add_argument("-v", "--video", type=int, default=0, help="Image - 0 , Video -1 ")
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
    "DICT_APRILTAG_36h11":cv2.aruco.DICT_APRILTAG_36h11
}

# Verify that the supplied ArUco tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"],None) is None:
    print(f"[INFO] ArUco tag of {args['type']} is not supported")
    sys.exit(0)

# load ArUco dictionary, grab the ArUco parameters, and detect the markers
print(f"[INFO] detecting {args['type']} tags...")
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()


def detect_aruco_markers(image):
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if len(corners)>0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the markker cornrs (which are always returned top-left, top-right, bottom-rigt, bottom-left order)
            corners = markerCorner.reshape(4,2)
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x,y) coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0,255,0),2)
            cv2.line(image,topRight, bottomRight, (0,255,0),2)
            cv2.line(image, bottomRight, bottomLeft, (0,255,0),2)
            cv2.line(image, bottomLeft, topLeft, (0,255,0), 2)

            cX = int((topLeft[0]+bottomRight[0])/2.0)
            cY = int((topLeft[1]+bottomRight[1])/2.0)
            cv2.circle(image, (cX,cY), 4, (0,0,255),-1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            print(f"[INFO] ArUco marker ID: {markerID}")

    cv2.imshow("Frame", image)




if args["video"]==0:
    print("[INFO] loading image...")
    image = cv2.imread(args["input"])
    image = imutils.resize(image, width=600)
    detect_aruco_markers(image)
else:
    vs = videostream(args["input"])
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        detect_aruco_markers(frame)

cv2.waitKey(0)
cv2.destroyAllWindows()






