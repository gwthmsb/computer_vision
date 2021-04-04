import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("chessboard.jpg")
sift = cv2.xfeatures2d.SIFT_create()
(kp, des) = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
plt.imshow(cv2.drawKeypoints(img, kp, None))
plt.show()