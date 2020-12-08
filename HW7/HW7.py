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
  
def thinning(image):
    height = 64
    width = 64
    change = True
    # 不再改變之後 break
    while change:
        change = False
        indices_p = []       
        # yokoi operator
        matrix_yokoi = drawing_Yokoi(downsample_image)
        # pair_relationship_operator
        for i in range(height):
            for j in range(width):
                if matrix_yokoi[i, j] == '1':
                    flag = False 
                    for element in NEIGHBORS:
                        if 0 <= i + element[0] < height and 0 <= j + element[1] < width:
                            if matrix_yokoi[i + element[0], j + element[1]] == '1':
                                flag = True
                                break
                    if flag:
                        indices_p.append((i, j))
        # compare marked_pixel with connected shrink operator
        for (i, j) in indices_p:
            count = 0
            for n, d in NEIGHBORS.items():
                if 0 <= i + n[0] < height and 0 <= j + n[1] < width:
                    neighbor = image[i + n[0], j + n[1]]
                    if neighbor:
                        h1, w1 = i + d[0][0], j + d[0][1]
                        h2, w2 = i + d[1][0], j + d[1][1]
                        # 只要超出範圍或是其中一個pixel不相等，則標註起來
                        if h1 < 0 or h1 == height or w1 < 0 or w1 == width or \
                            h2 < 0 or h2 == height or w2 < 0 or w2 == width:
                            count += 1
                        elif image[h1, w1] != neighbor or image[h2, w2] != neighbor:
                            count += 1
            # exactly one neighbor has yokoi number = 1
            if count == 1 and image[i, j] != 0:
                change = True
                image[i, j] = 0
    return image
          
if __name__ == '__main__':
    image = cv2.imread("lena.bmp")
    image_original = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            image_original[i][j] = image[i][j][0]
    binary_image = binary(image_original)
    downsample_image = downsampling(binary_image)
    
    NEIGHBORS = {
    (0, 1): [(-1, 0), (-1, 1)],     # a1 
    (-1, 0): [(-1, -1), (0, -1)],   # a2
    (0, -1): [(1, -1), (1, 0)],     # a3
    (1, 0): [(1, 1), (0, 1)]        # a4
    }

    thinning_image = thinning(downsample_image)   
    
    for i in range(64):
        for j in range(64):
            if thinning_image[i][j] == 1:
                thinning_image[i][j] = 255   
    cv2.imwrite("result.jpg", thinning_image)    
