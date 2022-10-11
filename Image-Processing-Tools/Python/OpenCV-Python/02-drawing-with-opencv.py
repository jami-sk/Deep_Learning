# Usgae
# python 02-drawing-with-opencv.py


import numpy as np
import cv2

canvas = np.zeros((300,300, 3), dtype=np.uint8)

# draw a diagonal line from TL to BR
green = (0,255,0)
cv2.line(canvas, (0,0), (300,300), green) #defualt one pixel width line
cv2.imshow("Green Line", canvas)
cv2.waitKey(0)

red = (0,0,255)
cv2.line(canvas, (300,0), (0,300), red, 2)
cv2.imshow("Green Line", canvas)
cv2.waitKey(0)

# draw a rectangle
cv2.rectangle(canvas, (50,50), (250,250), green, 2)
cv2.imshow("Green Square", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (100,100), (200,200), red, 3)
cv2.imshow("Red Square", canvas)
cv2.waitKey(0)

blue = (255,0,0)
cv2.rectangle(canvas, (125,125), (175,175) , blue, 3)
cv2.imshow("Red Square", canvas)
cv2.waitKey(0)


# draw a circle
white=(255,255,255)
(cX, cY) = (canvas.shape[1]//2, canvas.shape[0]//2)
for r in range(0,175,25):
    cv2.circle(canvas, (cX, cY), r, white)

cv2.imshow("Circles", canvas)
cv2.waitKey(0)


canvas = np.zeros((300,300,3), dtype="uint8")
for i in range(0,25):
    r = np.random.randint(5, 50)
    c = np.random.randint(0,255, size=(3,)).tolist()
    pt = np.random.randint(0,300, size=(2,))

    cv2.circle(canvas, tuple(pt), r, c, -1)

cv2.imshow("Random Circles", canvas)
cv2.waitKey(0)