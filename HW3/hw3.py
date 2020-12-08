import numpy as np
import cv2
import matplotlib.pyplot as plt

# original
img = cv2.imread("lena.bmp")
h,w,c = img.shape
hist_stat = np.zeros(256)
for i in range(h):
    for j in range(w):
        hist_stat[img[i][j]] += 1
histogram = {}
for i in range(256):
    histogram[i] = hist_stat[i]

plt.bar(histogram.keys(), histogram.values(), color='red')
plt.xlabel('Pixel value')
plt.ylabel('Amount')
plt.show()
print(h,w)

# divide intensity by 3
img_1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img_1[i][j] = np.round(img[i][j][0]/3)

cv2.imwrite("result_b.jpg", img_1)

hist_stat = np.zeros(256)
for i in range(h):
    for j in range(w):
        hist_stat[img_1[i][j]] += 1
histogram = {}
for i in range(256):
    histogram[i] = hist_stat[i]

plt.bar(histogram.keys(), histogram.values(), color='red')
plt.xlabel('Pixel value')
plt.ylabel('Amount')
plt.show()

# historam equalization
cdf = np.zeros(np.max(img_1)+1)
for i in range(np.max(img_1)+1):
    cdf[i] = np.sum(img_1 <= i)
print(np.min(cdf))

img_2 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img_2[i][j] = np.round((cdf[img_1[i][j]]-np.min(cdf))/(512*512-np.min(cdf))*255)
cv2.imwrite("result_c.jpg", img_2)

hist_stat = np.zeros(256)
for i in range(h):
    for j in range(w):
        hist_stat[img_2[i][j]] += 1
histogram = {}
for i in range(256):
    histogram[i] = hist_stat[i]

plt.bar(histogram.keys(), histogram.values(), color='red')
plt.xlabel('Pixel value')
plt.ylabel('Amount')
plt.show()