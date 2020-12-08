import numpy as np
import cv2
import matplotlib.pyplot as pyplot

img = cv2.imread("lena.bmp")
h,w,c = img.shape
print(h,w,c)

# upside-down
img_1 = np.zeros([h,w], dtype=np.uint8)
print(img_1.shape)
for i in range(h):
    for j in range(w):
        img_1[i][j] = img[511-i][j][0] 
cv2.imwrite("result_a.jpg",img_1)

# right-side-left
img_2 = np.zeros([h,w], dtype=np.uint8)
print(img_2.shape)
for i in range(h):
    for j in range(w):
        img_2[i][j] = img[i][511-j][0] 
cv2.imwrite("result_b.jpg",img_2)

# diagonally flip
img_3 = np.zeros([h,w], dtype=np.uint8)
print(img_3.shape)
for i in range(h):
    for j in range(w):
        img_3[i][j] = img[j][i][0] 
cv2.imwrite("result_c.jpg",img_3)