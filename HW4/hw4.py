import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp")
h,w,c = img.shape

# octogonal 35553 kernel
kernel = [[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-1],[2,0],[2,1]]
kernel_rev = [[-2,-2],[-2,2],[2,-2],[2,2]]

# J,K kernel
J_kernel = [[0,-1],[0,0],[1,0]]
K_kernel = [[-1,0],[-1,1],[0,1]]

img_b = np.zeros([h,w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if img[i][j][0] > 127:
            img_b[i][j] = 255
        else :
            img_b[i][j] = 0 

def dilation(img_b, kernel):
    res_1 = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if img_b[i][j] == 255:
                for element in kernel:
                    x, y = element
                    if i+x >= 0 and i+x <= h-1 and j+y >= 0 and j+y <= w-1:
                        res_1[i+x][j+y] = 255
    return res_1

def erosion(img_b, kernel):
    res_2 = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if img_b[i][j] == 255:
                for element in kernel:
                    x, y = element
                    if i+x < 0 or i+x > h-1 or j+y < 0 or j+y > w-1 or img_b[i+x][j+y] == 0:
                        res_2[i][j] = 0
                        break
                    if True :
                        res_2[i][j] = 255           
    return res_2           

def main():

    res_dil = dilation(img_b, kernel)
    cv2.imwrite("dilation.jpg", res_dil)

    res_ero = erosion(img_b, kernel)
    cv2.imwrite("erosion.jpg", res_ero)
    
    res_ope = dilation(erosion(img_b, kernel), kernel)
    cv2.imwrite("opening.jpg", res_ope)

    res_clo = erosion(dilation(img_b, kernel), kernel)
    cv2.imwrite("closing.jpg", res_clo)
    
    img_rev = -img_b + 255
    # res_ham = (((erosion(img_b, J_kernel) == 255) + (erosion(img_rev, K_kernel) == 255)) == 2) * 255
    res_ham = np.zeros([h,w], dtype=np.uint8)
    img_J = erosion(img_b, J_kernel)
    img_K = erosion(img_rev, K_kernel)
    for i in range(h):
        for j in range(w):
            if img_J[i][j]  == 255:
                if img_K[i][j] == 255:
                    res_ham[i][j] == 255
    cv2.imwrite("hitandmiss.jpg", res_ham)
if __name__ == '__main__':
    main()