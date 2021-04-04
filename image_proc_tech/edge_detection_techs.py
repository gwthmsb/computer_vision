from PIL import Image
from skimage.data import camera
import skimage.filters as filters
import skimage.feature as features
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

"""

# img = Image.open("giraffe.jpg")
img = Image.open("chessboard.jpg")
img_np = np.asarray(img)
img_gray = np.asarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))

img_edge_roberts = filters.roberts(img_gray)
img_edge_sobel = filters.sobel(img_gray)
img_edge_laplacian = filters.laplace(img_gray)
# plt.imshow(np.hstack((img_gray, img_edge_roberts, img_edge_sobel, img_edge_laplacian)), cmap='gray')
plt.imshow(img_edge_sobel, cmap='gray')
plt.show()
sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
plt.imshow(sobel, cmap='gray')
plt.show()

img_edge_cany = features.canny(img_gray, sigma=3)
plt.imshow(img_edge_cany, cmap='gray')
plt.show()