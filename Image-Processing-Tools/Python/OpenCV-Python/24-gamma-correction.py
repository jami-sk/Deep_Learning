#  USAGE
# python 24-gamma-correction.py -i <path to image>

import numpy as np
import argparse
import cv2


# Gamma correction is also called as Power Law Transform

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "path to input image")
args = vars(ap.parse_args())

original = cv2.imread(args["image"])

#loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
    if gamma==1:
        continue
    gamma = gamma if gamma>0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.putText(adjusted, f"g={gamma}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 3)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey(0)

