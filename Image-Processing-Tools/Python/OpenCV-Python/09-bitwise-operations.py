# Usage
# python 09-bitwise-operations.py

from cmath import rect
import cv2
import numpy as np

# Draw a rectable
rectangle = np.zeros((300,300), dtype=np.uint8)
cv2.rectangle(rectangle, (25,25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)

# Draw a circle
circle = np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (300//2, 300//2), 150, 255, -1)
cv2.imshow("Circle", circle)

#Bit Wise AND
bitwise_and = cv2.bitwise_and(rectangle, circle)
cv2.imshow("Bitwsie And", bitwise_and)

#Bit Wise OR
bitwise_or = cv2.bitwise_or(rectangle, circle)
cv2.imshow("Bitwsie Or", bitwise_or)

#Bitwise XOR
bitwise_xor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("Bitwsie Xor", bitwise_xor)

#Bitwise NOT
bitwise_not = cv2.bitwise_not(rectangle)
cv2.imshow("bitwise not", bitwise_not)
bitwise_not = cv2.bitwise_not(circle)
cv2.imshow("bitwise not", bitwise_not)
cv2.waitKey(0)