import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread("sudoku.png")
plt.imshow(img)
plt.show()

img_canny = cv2.Canny(img, 50, 200, None, 3)
plt.imshow(img_canny, cmap='gray')
plt.show()

lines = cv2.HoughLines(img_canny, 1, np.pi/180, 150, None, 0, 0)

print(lines)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
        cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

plt.imshow(img)
plt.show()