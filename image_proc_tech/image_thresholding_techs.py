from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


# importing image
img = Image.open("giraffe.jpg")
img_np = np.asarray(img)
print(np.shape(img_np))
# plt.imshow(img_np)
# plt.show()

img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
# plt.imshow(img_gray, cmap='gray')
# plt.show()

# ret: This is the threshold limit
# img_threshold: The image after applying threshold
ret, img_threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
i = np.hstack((img_gray, img_threshold))
# plt.imshow(i, cmap='gray')
# plt.show()

img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
i = np.hstack((img_gray, img_gaussian))
# plt.imshow(i, cmap='gray')
# plt.show()

hist, bins = np.histogram(img_gray.flatten(), 256, [0, 255])
# plt.plot(hist)
# plt.show()

# Adaptive threshold
img_adaptive_threshold1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
img_adaptive_threshold2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
img_adaptive_threshold3 = cv2.adaptiveThreshold(img_gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY, 11, 2)
# i = np.hstack((img_adaptive_threshold1, img_adaptive_threshold2, img_adaptive_threshold3))
# plt.imshow(i, cmap='gray')
# plt.show()


img_adaptive_threshold1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
img_adaptive_threshold2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 4)
i = np.hstack((img_adaptive_threshold1, img_adaptive_threshold2))
plt.imshow(i, cmap='gray')
plt.show()


# Otsu's threshold
ret, img_otsu = cv2.threshold(cv2.GaussianBlur(img_gray, (5, 5), 0), 0, 255, cv2.THRESH_OTSU)
plt.imshow(np.hstack((img_threshold, img_otsu)), cmap='gray')
plt.show()