import numpy as np
import cv2
import random
import hw5
from PIL import Image

def get_gaussian_image(img, ampli):
    h, w, c = img.shape
    img1 = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img1[i][j] = img[i][j][0]

    mean = 0
    var = 1
    gauss = np.random.normal(mean,var,(h,w))
    gauss = gauss.reshape(h,w)
    return img1 + ampli*gauss

def get_salt_and_pepper_image(img, prob):
    h, w, c = img.shape
    img1 = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            random_val = random.uniform(0,1)
            if random_val <= prob:
                img1[i][j] = 0
            elif random_val >= 1-prob:
                img1[i][j] = 255
            else:        
                img1[i][j] = img[i][j][0]
    return img1
       
def box_filter_3(img):
    h, w = img.shape
    img1 = np.zeros([h,w])
    for i in range(1,h-1):
        for j in range(1,w-1):          
            img1[i][j] = int(1/9*(int(img[i-1][j-1])+int(img[i-1][j])+int(img[i-1][j+1])+int(img[i][j-1])+int(img[i][j])+int(img[i][j+1])+int(img[i+1][j-1])+int(img[i+1][j])+int(img[i+1][j+1])))
    return img1

def box_filter_5(img):
    h, w = img.shape
    img1 = np.zeros([h,w])
    for i in range(2,h-2):
        for j in range(2,w-2):
            img1[i][j] = int(1/25*(int(img[i-2][j-2])+int(img[i-2][j-1])+int(img[i-2][j])+int(img[i-2][j+1])+int(img[i-2][j+2])+int(img[i-1][j-2])+int(img[i-1][j-1])+int(img[i-1][j])+int(img[i-1][j+1])+int(img[i-1][j+2])+int(img[i][j-2])+int(img[i][j-1])+int(img[i][j])+int(img[i][j+1])+int(img[i][j+2])+int(img[i+1][j-2])+int(img[i+1][j-1])+int(img[i+1][j])+int(img[i+1][j+1])+int(img[i+1][j+2])+int(img[i+2][j-2])+int(img[i+2][j-1])+int(img[i+2][j])+int(img[i+2][j+1])+int(img[i+2][j+2])))
    return img1

def median_filter_3(img):
    h, w = img.shape
    img1 = np.zeros([h,w])
    for i in range(1,h-1):
        for j in range(1,w-1):
            neighbor_list = [img[i-1][j-1],img[i-1][j],img[i-1][j+1],img[i][j-1],img[i][j],img[i][j+1],img[i+1][j-1],img[i+1][j],img[i+1][j+1]]
            neighbor_list.sort()
            img1[i][j] = neighbor_list[4]
    return img1 

def median_filter_5(img):
    h, w = img.shape
    img1 = np.zeros([h,w])
    for i in range(2,h-2):
        for j in range(2,w-2):
            neighbor_list = [img[i-2][j-2],img[i-2][j-1],img[i-2][j],img[i-2][j+1],img[i-2][j+2],img[i-1][j-2],img[i-1][j-1],img[i-1][j],img[i-1][j+1],img[i-1][j+2],img[i][j-2],img[i][j-1],img[i][j],img[i][j+1],img[i][j+2],img[i+1][j-2],img[i+1][j-1],img[i+1][j],img[i+1][j+1],img[i+1][j+2],img[i+2][j-2],img[i+2][j-1],img[i+2][j],img[i+2][j+1],img[i+2][j+2]]
            neighbor_list.sort()
            img1[i][j] = neighbor_list[12]
    return img1    

def opening(img):
    kernel = [[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-1],[2,0],[2,1]]
    opening_image = hw5.dilation(hw5.erosion(img, kernel), kernel)
    return opening_image

def closing(img):
    kernel = [[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-1],[2,0],[2,1]]
    closing_image = hw5.erosion(hw5.dilation(img, kernel), kernel)
    return closing_image

def SNR(signal_img, noise_img):
    h, w = signal_img.shape
    total = 0
    for i in range(h):
        for j in range(w):
            total += signal_img[i][j]
    mu = total/(h*w)
    
    VS_top = 0
    for i in range(h):
        for j in range(w):
            VS_top += np.square(signal_img[i][j]-mu)
    VS = VS_top/(h*w)
    
    total_noise = 0
    for i in range(h):
        for j in range(w):
            total_noise += (noise_img[i][j]-signal_img[i][j])
    mu_noise = total_noise/(h*w)
    
    VN_top = 0
    for i in range(h):
        for j in range(w):
            VN_top += np.square(noise_img[i][j]-signal_img[i][j]-mu_noise) 
    VN = VN_top/(h*w)
    
    return 20*np.log10(np.sqrt(VS)/np.sqrt(VN))

if __name__ == '__main__':
     
    original_image = cv2.imread("lena.bmp")
    ori_img = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            ori_img[i][j] = original_image[i][j][0]
    # 產生 gaussian noise
    gaussian_10 = get_gaussian_image(original_image, 10)
    gaussian_30 = get_gaussian_image(original_image, 30)
    cv2.imwrite("gaussian_10/gaussian_10.jpg", gaussian_10) 
    cv2.imwrite("gaussian_30/gaussian_30.jpg", gaussian_30) 
    gaussian_10_SNR = SNR(ori_img, gaussian_10)
    gaussian_30_SNR = SNR(ori_img, gaussian_30)
    
    # 產生salt-and-pepper noise
    salt_and_pepper_01 = get_salt_and_pepper_image(original_image, 0.1)
    salt_and_pepper_005 = get_salt_and_pepper_image(original_image, 0.05)
    cv2.imwrite("salt_and_pepper_0.1/salt_and_pepper_0.1.jpg", salt_and_pepper_01)  
    cv2.imwrite("salt_and_pepper_0.05/salt_and_pepper_0.05.jpg", salt_and_pepper_005)
    salt_and_pepper_01_SNR = SNR(ori_img, salt_and_pepper_01)
    salt_and_pepper_005_SNR = SNR(ori_img, salt_and_pepper_005)

    # 3*3 box filter
    gaussian_10_box_3 = box_filter_3(gaussian_10)
    gaussian_30_box_3 = box_filter_3(gaussian_30)
    salt_and_pepper_01_box_3 = box_filter_3(salt_and_pepper_01)
    salt_and_pepper_005_box_3 = box_filter_3(salt_and_pepper_005)
    
    gaussian_10_box_3_SNR = SNR(ori_img, gaussian_10_box_3)
    gaussian_30_box_3_SNR = SNR(ori_img, gaussian_30_box_3)
    salt_and_pepper_01_box_3_SNR = SNR(ori_img, salt_and_pepper_01_box_3)
    salt_and_pepper_005_box_3_SNR = SNR(ori_img, salt_and_pepper_005_box_3)
    
    cv2.imwrite("gaussian_10/gaussian_10_box_3.jpg", gaussian_10_box_3)
    cv2.imwrite("gaussian_30/gaussian_30_box_3.jpg", gaussian_30_box_3)
    cv2.imwrite("salt_and_pepper_0.1/salt_and_pepper_0.1_box_3.jpg", salt_and_pepper_01_box_3)
    cv2.imwrite("salt_and_pepper_0.05/salt_and_pepper_0.05_box_3.jpg", salt_and_pepper_005_box_3)
    

    # 5*5 box filter
    gaussian_10_box_5 = box_filter_5(gaussian_10)
    gaussian_30_box_5 = box_filter_5(gaussian_30)
    salt_and_pepper_01_box_5 = box_filter_5(salt_and_pepper_01)
    salt_and_pepper_005_box_5 = box_filter_5(salt_and_pepper_005)
    
    gaussian_10_box_5_SNR = SNR(ori_img, gaussian_10_box_5)
    gaussian_30_box_5_SNR = SNR(ori_img, gaussian_30_box_5)
    salt_and_pepper_01_box_5_SNR = SNR(ori_img, salt_and_pepper_01_box_5)
    salt_and_pepper_005_box_5_SNR = SNR(ori_img, salt_and_pepper_005_box_5)
    
    cv2.imwrite("gaussian_10/gaussian_10_box_5.jpg", gaussian_10_box_5)
    cv2.imwrite("gaussian_30/gaussian_30_box_5.jpg", gaussian_30_box_5)
    cv2.imwrite("salt_and_pepper_0.1/salt_and_pepper_0.1_box_5.jpg", salt_and_pepper_01_box_5)
    cv2.imwrite("salt_and_pepper_0.05/salt_and_pepper_0.05_box_5.jpg", salt_and_pepper_005_box_5)
    
    
    # 3*3 median filter
    gaussian_10_median_3 = median_filter_3(gaussian_10)
    gaussian_30_median_3 = median_filter_3(gaussian_30)
    salt_and_pepper_01_median_3 = median_filter_3(salt_and_pepper_01)
    salt_and_pepper_005_median_3 = median_filter_3(salt_and_pepper_005)

    gaussian_10_median_3_SNR = SNR(ori_img, gaussian_10_median_3)
    gaussian_30_median_3_SNR = SNR(ori_img, gaussian_30_median_3)
    salt_and_pepper_01_median_3_SNR = SNR(ori_img, salt_and_pepper_01_median_3)
    salt_and_pepper_005_median_3_SNR = SNR(ori_img, salt_and_pepper_005_median_3)

    cv2.imwrite("gaussian_10/gaussian_10_median_3.jpg", gaussian_10_median_3)
    cv2.imwrite("gaussian_30/gaussian_30_median_3.jpg", gaussian_30_median_3)
    cv2.imwrite("salt_and_pepper_0.1/salt_and_pepper_0.1_median_3.jpg", salt_and_pepper_01_median_3)
    cv2.imwrite("salt_and_pepper_0.05/salt_and_pepper_0.05_median_3.jpg", salt_and_pepper_005_median_3)
    
    # 5*5 median filter
    gaussian_10_median_5 = median_filter_5(gaussian_10)
    gaussian_30_median_5 = median_filter_5(gaussian_30)
    salt_and_pepper_01_median_5 = median_filter_5(salt_and_pepper_01)
    salt_and_pepper_005_median_5 = median_filter_5(salt_and_pepper_005)
    
    gaussian_10_median_5_SNR = SNR(ori_img, gaussian_10_median_5)
    gaussian_30_median_5_SNR = SNR(ori_img, gaussian_30_median_5)
    salt_and_pepper_01_median_5_SNR = SNR(ori_img, salt_and_pepper_01_median_5)
    salt_and_pepper_005_median_5_SNR = SNR(ori_img, salt_and_pepper_005_median_5)

    cv2.imwrite("gaussian_10/gaussian_10_median_5.jpg", gaussian_10_median_5)
    cv2.imwrite("gaussian_30/gaussian_30_median_5.jpg", gaussian_30_median_5)
    cv2.imwrite("salt_and_pepper_0.1/salt_and_pepper_0.1_median_5.jpg", salt_and_pepper_01_median_5)
    cv2.imwrite("salt_and_pepper_0.05/salt_and_pepper_0.05_median_5.jpg", salt_and_pepper_005_median_5)

    # opening then closing & closing then opening
    gaussian_10 = Image.open("gaussian_10/gaussian_10.jpg")
    result_image = closing(opening(gaussian_10))
    result1_image = opening(closing(gaussian_10))
    result_image.save("gaussian_10/gaussian_10_openingthenclosing.bmp")
    result1_image.save("gaussian_10/gaussian_10_closingthenopening.bmp")
    
    or1 = cv2.imread("gaussian_10/gaussian_10.jpg")
    or2 = cv2.imread("gaussian_10/gaussian_10_openingthenclosing.bmp")
    or3 = cv2.imread("gaussian_10/gaussian_10_closingthenopening.bmp")
    gaussian_10 = np.zeros([512,512])
    gaussian_10_openingthenclosing = np.zeros([512,512])
    gaussian_10_closingthenopening = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            gaussian_10[i][j] = or1[i][j][0]
            gaussian_10_openingthenclosing[i][j] = or2[i][j][0]
            gaussian_10_closingthenopening[i][j] = or3[i][j][0]
    gaussian_10_openingthenclosing_SNR = SNR(ori_img, gaussian_10_openingthenclosing)
    gaussian_10_closingthenopening_SNR = SNR(ori_img, gaussian_10_closingthenopening)

    gaussian_30 = Image.open("gaussian_30/gaussian_30.jpg")
    result_image = closing(opening(gaussian_30))
    result1_image = opening(closing(gaussian_30))
    result_image.save("gaussian_30/gaussian_30_openingthenclosing.bmp")
    result1_image.save("gaussian_30/gaussian_30_closingthenopening.bmp")
 
    or1 = cv2.imread("gaussian_30/gaussian_30.jpg")
    or2 = cv2.imread("gaussian_30/gaussian_30_openingthenclosing.bmp")
    or3 = cv2.imread("gaussian_30/gaussian_30_closingthenopening.bmp")
    gaussian_30 = np.zeros([512,512])
    gaussian_30_openingthenclosing = np.zeros([512,512])
    gaussian_30_closingthenopening = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            gaussian_30[i][j] = or1[i][j][0]
            gaussian_30_openingthenclosing[i][j] = or2[i][j][0]
            gaussian_30_closingthenopening[i][j] = or3[i][j][0]

    gaussian_30_openingthenclosing_SNR = SNR(ori_img, gaussian_30_openingthenclosing)
    gaussian_30_closingthenopening_SNR = SNR(ori_img, gaussian_30_closingthenopening)

    salt_and_pepper_01 = Image.open("salt_and_pepper_0.1/salt_and_pepper_0.1.jpg")
    result_image = closing(opening(salt_and_pepper_01))
    result1_image = opening(closing(salt_and_pepper_01))
    result_image.save("salt_and_pepper_0.1/salt_and_pepper_0.1_openingthenclosing.bmp")
    result1_image.save("salt_and_pepper_0.1/salt_and_pepper_0.1_closingthenopening.bmp")
 
    or1 = cv2.imread("salt_and_pepper_0.1/salt_and_pepper_0.1.jpg")
    or2 = cv2.imread("salt_and_pepper_0.1/salt_and_pepper_0.1_openingthenclosing.bmp")
    or3 = cv2.imread("salt_and_pepper_0.1/salt_and_pepper_0.1_closingthenopening.bmp")
    salt_and_pepper_01 = np.zeros([512,512])
    salt_and_pepper_01_openingthenclosing = np.zeros([512,512])
    salt_and_pepper_01_closingthenopening = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            salt_and_pepper_01[i][j] = or1[i][j][0]
            salt_and_pepper_01_openingthenclosing[i][j] = or2[i][j][0]
            salt_and_pepper_01_closingthenopening[i][j] = or3[i][j][0]

    salt_and_pepper_01_openingthenclosing_SNR = SNR(ori_img, salt_and_pepper_01_openingthenclosing)
    salt_and_pepper_01_closingthenopening_SNR = SNR(ori_img, salt_and_pepper_01_closingthenopening)

    salt_and_pepper_005 = Image.open("salt_and_pepper_0.05/salt_and_pepper_0.05.jpg")
    result_image = closing(opening(salt_and_pepper_005))
    result1_image = opening(closing(salt_and_pepper_005))
    result_image.save("salt_and_pepper_0.05/salt_and_pepper_0.05_openingthenclosing.bmp")
    result1_image.save("salt_and_pepper_0.05/salt_and_pepper_0.05_closingthenopening.bmp")
    
    or1 = cv2.imread("salt_and_pepper_0.05/salt_and_pepper_0.05.jpg")
    or2 = cv2.imread("salt_and_pepper_0.05/salt_and_pepper_0.05_openingthenclosing.bmp")
    or3 = cv2.imread("salt_and_pepper_0.05/salt_and_pepper_0.05_closingthenopening.bmp")
    salt_and_pepper_005 = np.zeros([512,512])
    salt_and_pepper_005_openingthenclosing = np.zeros([512,512])
    salt_and_pepper_005_closingthenopening = np.zeros([512,512])
    for i in range(512):
        for j in range(512):
            salt_and_pepper_005[i][j] = or1[i][j][0]
            salt_and_pepper_005_openingthenclosing[i][j] = or2[i][j][0]
            salt_and_pepper_005_closingthenopening[i][j] = or3[i][j][0]

    salt_and_pepper_005_openingthenclosing_SNR = SNR(ori_img, salt_and_pepper_005_openingthenclosing)
    salt_and_pepper_005_closingthenopening_SNR = SNR(ori_img, salt_and_pepper_005_closingthenopening)
    
    # 將SNR結果寫出
    file = open("SNR.txt", "w")
    file.write("gaussian_10_SNR: " + str(gaussian_10_SNR) + '\n')
    file.write("gaussian_10_box_3_SNR: " + str(gaussian_10_box_3_SNR) + '\n')
    file.write("gaussian_10_box_5_SNR: " + str(gaussian_10_box_5_SNR) + '\n')
    file.write("gaussian_10_median_3_SNR: " + str(gaussian_10_median_3_SNR) + '\n')
    file.write("gaussian_10_median_5_SNR: " + str(gaussian_10_median_5_SNR) + '\n')
    file.write("gaussian_10_openingthenclosing_SNR: " + str(gaussian_10_openingthenclosing_SNR) + '\n')
    file.write("gaussian_10_closingthenopening_SNR: " + str(gaussian_10_closingthenopening_SNR) + '\n')
    file.write("------------------------------------------------ \n")
    file.write("gaussian_30_SNR: " + str(gaussian_30_SNR) + '\n')
    file.write("gaussian_30_box_3_SNR: " + str(gaussian_30_box_3_SNR) + '\n')
    file.write("gaussian_30_box_5_SNR: " + str(gaussian_30_box_5_SNR) + '\n')
    file.write("gaussian_30_median_3_SNR: " + str(gaussian_30_median_3_SNR) + '\n')
    file.write("gaussian_30_median_5_SNR: " + str(gaussian_30_median_5_SNR) + '\n')
    file.write("gaussian_30_openingthenclosing_SNR: " + str(gaussian_30_openingthenclosing_SNR) + '\n')
    file.write("gaussian_30_closingthenopening_SNR: " + str(gaussian_30_closingthenopening_SNR) + '\n')
    file.write("------------------------------------------------ \n")
    file.write("salt_and_pepper_01_SNR: " + str(salt_and_pepper_01_SNR) + '\n')
    file.write("salt_and_pepper_01_box_3_SNR: " + str(salt_and_pepper_01_box_3_SNR) + '\n')
    file.write("salt_and_pepper_01_box_5_SNR: " + str(salt_and_pepper_01_box_5_SNR) + '\n')
    file.write("salt_and_pepper_01_median_3_SNR: " + str(salt_and_pepper_01_median_3_SNR) + '\n')
    file.write("salt_and_pepper_01_median_5_SNR: " + str(salt_and_pepper_01_median_5_SNR) + '\n')
    file.write("salt_and_pepper_01_openingthenclosing_SNR: " + str(salt_and_pepper_01_openingthenclosing_SNR) + '\n')
    file.write("salt_and_pepper_01_closingthenopening_SNR: " + str(salt_and_pepper_01_closingthenopening_SNR) + '\n')
    file.write("------------------------------------------------ \n")
    file.write("salt_and_pepper_005_SNR: " + str(salt_and_pepper_005_SNR) + '\n')
    file.write("salt_and_pepper_005_box_3_SNR: " + str(salt_and_pepper_005_box_3_SNR) + '\n')
    file.write("salt_and_pepper_005_box_5_SNR: " + str(salt_and_pepper_005_box_5_SNR) + '\n')
    file.write("salt_and_pepper_005_median_3_SNR: " + str(salt_and_pepper_005_median_3_SNR) + '\n')
    file.write("salt_and_pepper_005_median_5_SNR: " + str(salt_and_pepper_005_median_5_SNR) + '\n')
    file.write("salt_and_pepper_005_openingthenclosing_SNR: " + str(salt_and_pepper_005_openingthenclosing_SNR) + '\n')
    file.write("salt_and_pepper_005_closingthenopening_SNR: " + str(salt_and_pepper_005_closingthenopening_SNR) + '\n')