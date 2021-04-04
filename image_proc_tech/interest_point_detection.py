import cv2
import matplotlib.pyplot as plt
import numpy as np


"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html

"""


img = cv2.imread("chessboard.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray[0:10][0:10])
img_gray = np.float32(img_gray)
# print(img_gray[0:10][0:10])

"""
Harris corner detector
Arguments: 
    img: Input image, it should be grayscale and float32 type
    blockSize: It is the size of neighbourhood considered for corner detection
    ksize: Aperture parameter for sobel derivative used
    k: Harris detector free parameter in the equation 

"""
dst = cv2.cornerHarris(img_gray, 3, 3, 0.04)
plt.imshow(dst, cmap='gray')
# plt.imshow(cv2.dilate(dst, None),  cmap='gray')
plt.show()
print(dst.max())
print(0.01*dst.max())

print(dst>0.01*dst.max())
img[dst>0.01*dst.max()] = [255, 0, 0]
print(img)
plt.imshow(img)
plt.show()

corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x,y), 20, 255, -1)
plt.imshow(img)
plt.show()