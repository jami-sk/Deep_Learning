# USAGE
# python 26-detecting-low-contrast-images.py -i <path to image>

from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import cv2
import imutils
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help = "path to input directory of images")
ap.add_argument("-t","--thresh", type=float, default = 0.35,help="threshold for low contrast")
ap.add_argument("-v", "--video", default=0, help="Path to the input video file. Dfault webcame (0)")
args = vars(ap.parse_args())

if args["video"] ==0:
    imagePaths = sorted(list(list_images(args["input"])))

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] processing image {i+1}/{len(imagePaths)}")
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=450)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        text = "Low Contrast: No"
        color = (0,255,0)

        if is_low_contrast(gray, fraction_threshold=args["thresh"]):
            text = "Low Contrast: Yes"
            color = (0,0, 255)
        else:
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(image, [c],-1, (0,255,0), 2)

        cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Image", image)
        cv2.imshow("Edge", edged)
        cv2.waitKey(0)
else:
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(args["video"])
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("[INFO] no frames read from stream - exiting")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        text = "Low Contrast: No"
        color = (0,255,0)

        if is_low_contrast(gray, fraction_threshold=args["thresh"]):
            text = "Low Contrast: Yes"
            color = (0,0, 255)
        else:
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(frame, [c],-1, (0,255,0), 2)

        cv2.putText(frame, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        output = np.dstack([edged]*3)
        output = np.hstack([frame, output])

        cv2.imshow("Output", output)
        key = cv2.waitKey(1)&0XFF
        if key== ord("q"):
            break
