import numpy as np
import cv2
import matplotlib.pyplot as plt

# binary image
img = cv2.imread("lena.bmp")
h,w,c = img.shape
img_1 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if img[i][j][0] > 127:
            img_1[i][j] = 255
        else :
            img_1[i][j] = 0    
cv2.imwrite("result_a.jpg",img_1)

# histogram
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

# connected component
img_2 = cv2.imread("result_a.jpg")
number, label, stats, centers = cv2.connectedComponentsWithStats(img_1, connectivity=8)

for i in range(1,number):
    if stats[i][4] > 500:
        start_x, start_y , width, height, area = stats[i]
        x, y = centers[i]

        cv2.circle(img_2, (np.int(x), np.int(y)), 2, (0,0,255), 3, 8, 0)
        cv2.rectangle(img_2, (start_x, start_y), (start_x+width, start_y+height), (225, 105, 65), 2, 8, 0)

# cv2.imshow("colored labels", img_2)
cv2.imwrite("result_c.jpg",img_2)