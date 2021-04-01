import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

image = Image.open('lena.png')
#plt.imshow(image)
#plt.show()

print(np.shape(image))

image1 = np.asarray(image)
#print(len(image1))
# plt.imshow(image1[:,:,0], cmap='gray')
# plt.show()
# plt.imshow(image1[:,:,1], cmap='gray')
# plt.show()
# plt.imshow(image1[:,:,2], cmap='gray')
# plt.show()

#print(image1[1:10, 1:10, 1])
# plt.imshow(image1[1:10, 1:10])
# plt.show()


# Histogram from a gray scale image
image_cv2 = cv2.imread("lena.png")
#print(image_cv2)
# channel:0 is blue. Basically BGR, histogram for channel
histogram_image = cv2.calcHist(images=[image_cv2], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
#print(histogram_image)
plt.plot(histogram_image)
# plt.show()

#Image histogram
plt.hist(image_cv2.ravel(), 256, [0, 256])
plt.show()

print(np.mean(histogram_image), np.mean(image))
print(np.median(histogram_image), np.median(image))
print(np.max(histogram_image))
print(np.min(histogram_image))
print(np.var(histogram_image))
print(np.std(histogram_image))