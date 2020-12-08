import numpy as np
import cv2

def Laplacian_1(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])

    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = 0*img1[i-1][j-1]+img1[i-1][j]+0*img1[i-1][j+1]+img1[i][j-1]+(-4)*img1[i][j]+img1[i][j+1]+0*img1[i+1][j-1]+img1[i+1][j]+0*img1[i+1][j+1]
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 1
            elif result[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0 
    return result

def Laplacian_2(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])

    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = 1/3*(img1[i-1][j-1]+img1[i-1][j]+img1[i-1][j+1]+img1[i][j-1]+(-8)*img1[i][j]+img1[i][j+1]+img1[i+1][j-1]+img1[i+1][j]+img1[i+1][j+1])
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 1
            elif result[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0 
    return result    

def minimum_var_Laplacian(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])

    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = 1/3*(2*img1[i-1][j-1]-img1[i-1][j]+2*img1[i-1][j+1]-img1[i][j-1]+(-4)*img1[i][j]-img1[i][j+1]+2*img1[i+1][j-1]-img1[i+1][j]+2*img1[i+1][j+1])
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 1
            elif result[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0 
    return result


def Laplacian_of_Gaussian(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])

    kernel = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
			  [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
			  [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
			  [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
			  [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
			  [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
			  [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
			  [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
			  [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
			  [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
			  [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]

    for i in range(5,h-5):
        for j in range(5,w-5):
            neighbors = np.zeros([11,11])
            for x in range(11):
                for y in range(11):
                    neighbors[x][y] = img1[i-5+x][j-5+y]
            for x in range(11):
                for y in range(11):
                    result[i][j] += kernel[x][y]*neighbors[x][y]       
   
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 1
            elif result[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0 
    return result

def Difference_of_Gaussian(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])

    kernel = [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
			 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
			 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
			 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
			 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
			 [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
			 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
			 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
			 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
			 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
			 [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]

    for i in range(5,h-5):
        for j in range(5,w-5):
            neighbors = np.zeros([11,11])
            for x in range(11):
                for y in range(11):
                    neighbors[x][y] = img1[i-5+x][j-5+y]
            for x in range(11):
                for y in range(11):
                    result[i][j] += kernel[x][y]*neighbors[x][y]       
   
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 1
            elif result[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0    
    return result

def zero_crossing_edge_detector(original_img):
    h, w = original_img.shape
    result = np.zeros([h,w])
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = 255
            if original_img[i][j] == 1:
                for x in range(3):
                    for y in range(3):
                        if original_img[i-1+x][j-1+y] == -1:
                            result[i][j] = 0
    return result


if __name__ == "__main__" :
    original_img = cv2.imread("lena.bmp")

    Laplacian_1_img = Laplacian_1(original_img, 15)               
    Laplacian_1_img = zero_crossing_edge_detector(Laplacian_1_img)
    cv2.imwrite("Laplacian_1.jpg", Laplacian_1_img)

    Laplacian_2_img = Laplacian_2(original_img, 15)
    Laplacian_2_img = zero_crossing_edge_detector(Laplacian_2_img)
    cv2.imwrite("Laplacian_2.jpg", Laplacian_2_img)

    minimum_var_Laplacian_img = minimum_var_Laplacian(original_img, 20)
    minimum_var_Laplacian_img = zero_crossing_edge_detector(minimum_var_Laplacian_img)
    cv2.imwrite("minimum_var_Laplacian.jpg", minimum_var_Laplacian_img)

    Laplacian_of_Gaussian_img = Laplacian_of_Gaussian(original_img, 3000)
    Laplacian_of_Gaussian_img = zero_crossing_edge_detector(Laplacian_of_Gaussian_img)
    cv2.imwrite("Laplacian_of_Gaussian.jpg", Laplacian_of_Gaussian_img)

    Difference_of_Gaussian_img = Difference_of_Gaussian(original_img, 1)
    Difference_of_Gaussian_img = zero_crossing_edge_detector(Difference_of_Gaussian_img)
    cv2.imwrite("Difference_of_Gaussian.jpg", Difference_of_Gaussian_img)