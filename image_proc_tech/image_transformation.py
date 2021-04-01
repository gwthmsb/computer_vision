from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = Image.open('lena.png')
print(img) # Image object

img_np = np.asarray(img)
print(np.shape(img_np)) # Transformed image as array

grayscale = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
img_cv = np.asarray(grayscale)
print(img_cv.shape)

# Difference between original image and grayscale
print(img_np[0][0][0:])
print(img_cv[0][0])

#Histogram equalization

img_e = cv2.equalizeHist(img_cv)
# plt.imshow(img_cv)
# plt.show()
# plt.imshow(equ)
# plt.show()
print(img_e[0][0])
img_stack = np.hstack((img_cv, img_e))
# plt.imshow(img_stack, cmap='gray')
# plt.show()

#Histogram difference
# plt.hist(img_cv.ravel(), 256, [0, 256])
# plt.hist(img_e.ravel(), 256, [0, 256])
# plt.show()

# Cumulative distribution function
hist, bins = np.histogram(img_cv.flatten(), 256, [0, 256])
# print(hist) # hist: y-axis value of x-axis discrete ele in histogram
# print(bins) # bins: x-axis discrete elements
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
# print(cdf_normalized)
# i = np.hstack((cdf, cdf_normalized))
# plt.plot(cdf)
# plt.show()
# plt.plot(cdf_normalized)
# plt.show()

# Cumulative distribution function of histogram equalized image
hist_e, bin_e = np.histogram(img_e.flatten(), 256, [0, 256])
cdf_e = hist_e.cumsum()
cdf_e_norm = cdf_e * hist_e.max() / cdf_e.max()
# plt.plot(cdf_e_norm)
# plt.show()


# Gaussian filter

kernel = np.ones((5, 5), np.float32)/25
print(kernel)
img_filtered = cv2.filter2D(img_cv, -1, kernel)
img_filtered_gau = cv2.GaussianBlur(img_cv, (5,5), 0)
# i = np.hstack((img_cv, img_filtered, img_filtered_gau))
# plt.imshow(i, cmap='gray')
# plt.show()

# FFT of image
# 2 dimensional FFT
fftOfImg = np.fft.fft2(img_cv)
# print(img_cv[0:4, 0:4])
# print(fftOfImg[0:4, 0:4])
print(np.shape(fftOfImg))
fftOfImg[400:511, 400:511] = 0
ifftOfImg = np.fft.ifft2(fftOfImg)
i = np.hstack((img_cv, ifftOfImg))
plt.imshow(np.real(i), cmap='gray')
plt.show()

fshift = np.fft.fftshift(fftOfImg)
magnitude_spectrum = 20*np.log(np.abs(fshift+1))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()