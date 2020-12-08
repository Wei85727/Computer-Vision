import cv2
import numpy as np
from PIL import Image


def binary(image_original):
    for i in range(512):
        for j in range(512):
            if image_original[i][j] >= 128:
                image_original[i][j] = 1
            else:
                image_original[i][j] = 0
    return image_original

def downsampling(binary_image):
    downsample_image = np.zeros([64,64])
    for i in range(64):
        for j in range(64):
            downsample_image[i][j] = binary_image[8*i][8*j]
    return downsample_image

def neighborpixel(image, pos):
    kernel = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]   
    neighborpixels = np.zeros(9)
    x, y = pos
    for element in kernel:
        i, j = element
        if 0 <= x+i < 64 and 0 <= y+j < 64:
            neighborpixels[(i+1)+3*(j+1)] = image[x+i][y+j]  
        else :
            neighborpixels[(i+1)+3*(j+1)] = 0    
    neighborpixels1 = [neighborpixels[4], neighborpixels[5], neighborpixels[1], neighborpixels[3], neighborpixels[7], neighborpixels[8], neighborpixels[2], neighborpixels[0], neighborpixels[6]]
    
    return neighborpixels1        

def function_h(b, c, d, e):
    if b == c and (b != d or b != e):
        return("q")
    if b == c and (b == d and b == e):
        return("r")
    if b != c:
        return("s")

def function_f(a1, a2, a3, a4):
    count_list = [a1, a2, a3, a4]
    if count_list.count("r") == 4:  
        return(5)
    else :
        return(count_list.count("q"))    

def drawing_Yokoi(downsample_image):
    Yokoi_Connectivity_Number = np.full([64,64], " ")
    for i in range(64):
        for j in range(64):
            if downsample_image[i][j] == 1:
                neighborpixels = neighborpixel(downsample_image, (i, j))
                Yokoi_Connectivity_Number[i][j] = function_f(
                    function_h(neighborpixels[0], neighborpixels[1], neighborpixels[6], neighborpixels[2]),
                    function_h(neighborpixels[0], neighborpixels[2], neighborpixels[7], neighborpixels[3]),
                    function_h(neighborpixels[0], neighborpixels[3], neighborpixels[8], neighborpixels[4]), 
                    function_h(neighborpixels[0], neighborpixels[4], neighborpixels[5], neighborpixels[1]))
            else:
                Yokoi_Connectivity_Number[i][j] = " "        
    return Yokoi_Connectivity_Number
  
if __name__ == '__main__':
    image = cv2.imread("lena.bmp")
    image_original = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            image_original[i][j] = image[i][j][0]
    binary_image = binary(image_original)
    downsample_image = downsampling(binary_image)
    # cv2.imwrite("binary.bmp", binary_image)
    # cv2.imwrite("downsample.png", downsample_image)
    Yokoi_Connectivity_Number = drawing_Yokoi(downsample_image)
    # delimiter由預設的留一空格改成不留空
    np.savetxt("Yokoi_result.txt", Yokoi_Connectivity_Number, delimiter="", fmt='%s') 
    # a = neighborpixel(downsample_image, (0,0))
    # b = funtion_h(1,1,0,1)
    a = 0
    b = 0
    for i in range(64):
        for j in range(64):
            if downsample_image[i][j] == 1:
                a += 1
            else:
                b += 1     
    print(a,b,a+b,64*64)            
         