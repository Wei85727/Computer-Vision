import numpy as np
import cv2
import matplotlib.pyplot as pyplot
from PIL import Image

img = cv2.imread("lena.bmp")
h,w,c = img.shape
print(h,w,c)

# rotate 45 clockwise
(X, Y) = (w/2, h/2)
M = cv2.getRotationMatrix2D((X, Y), -45, 1)
image = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("result_d.jpg",image)

# shrink half
img_test = Image.open("lena.bmp")
print(img_test.size)

img_2 = img_test.resize((256,256))
print(img_2.size)

img_2.save("result_e.jpg")

# binarize 
img = cv2.imread("lena.bmp")
img_3 = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if img[i][j][0] > 128:
            img_3[i][j] = 255
        else :
            img_3[i][j] = 0    
cv2.imwrite("result_f.jpg",img_3)