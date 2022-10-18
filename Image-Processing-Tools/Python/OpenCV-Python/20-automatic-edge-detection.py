# Usage
# python 20-automatic-edge-detection.py

import numpy as np
import argparse
import glob, cv2

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))

    edged = cv2.Canny(image, lower, upper)

    return edged

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help = "path to input dataset of images")
args = vars(ap.parse_args())

for imagePath in glob.glob(args["images"]+"/*.jpg"):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    wide = cv2.Canny(blurred, 10,200)
    mid = cv2.Canny(blurred, 50,150)
    tight = cv2.Canny(blurred, 225,250)
    auto = auto_canny(blurred)

    cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide,mid,tight,auto]))
    cv2.waitKey(0)


