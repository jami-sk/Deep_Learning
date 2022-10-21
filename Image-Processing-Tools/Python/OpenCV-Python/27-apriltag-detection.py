# USAGE
# python 27-apriltag-detection.py

import cv2
import argparse
import apriltag

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

print("[INFO] Loading image ...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO] detecting AprilTags...")
options = apriltag.DetectotOptions(families="yshd") # "tag36h11" "tagstandard41h12" "tagstandard52h13" "tagcircle21h7" "tagcircle49h12" "tagcustom48h12"
detector = apriltag.Detector(options)
results = detector.detect(gray)

print(f"[INFO] {len(results)} total AprilTags detected")

for r in results:
    (ptA, ptB, ptC, ptD) = r.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    cv2.line(image, ptA, ptB, (0,255,0), 2)
    cv2.line(image, ptB, ptC, (0,255,0), 2)
    cv2.line(image, ptC, ptD, (0,255,0), 2)
    cv2.line(image, ptD, ptA, (0,255,0), 2)

    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv2.circle(image, (cX,cY), 5, (0,0,255 -1))

    tagFamily = r.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    print(f"[INFO] tag family: {tagFamily}")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



