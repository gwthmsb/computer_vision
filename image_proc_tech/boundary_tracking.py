import numpy as np
import skimage.draw as draw
import skimage.measure as measure
import skimage.transform as transform
import skimage.feature as feature
import matplotlib.pyplot as plt
import cv2

img = np.zeros((600, 600))
# plt.imshow(img, cmap='gray')
# plt.show()

rr, cc = draw.ellipse(300, 350, 100, 220)
print(rr, cc)
img[rr, cc] = 1
# plt.imshow(img, cmap='gray')
# plt.show()

# For fun. Border detection
img_canny = feature.canny(img)
# plt.imshow(img_canny, cmap='gray')
# plt.show()

img_rotate = transform.rotate(img, angle=20, order=0)
plt.imshow(img_rotate, cmap='gray')
plt.show()