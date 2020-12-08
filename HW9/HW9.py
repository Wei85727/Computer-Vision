import numpy as np
import cv2

def Roberts(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    for i in range(h-1):
        for j in range(w-1):
            img1[i][j] = np.sqrt(np.power((img1[i+1][j+1]-img1[i][j]), 2)+np.power((img1[i+1][j]-img1[i][j+1]), 2))
    for i in range(h):
        for j in range(w):  
            if img1[i][j] >= threshold:
                img1[i][j] = 0
            else:
                img1[i][j] = 255
    return img1            
                
def Prewitt(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = np.sqrt(np.power((img1[i+1][j-1]+img1[i+1][j]+img1[i+1][j+1]-img1[i-1][j-1]-img1[i-1][j]-img1[i-1][j+1]), 2)+np.power((img1[i-1][j+1]+img1[i][j+1]+img1[i+1][j+1]-img1[i-1][j-1]-img1[i][j-1]-img1[i+1][j-1]), 2))
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result 

def Sobel(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = np.sqrt(np.power((img1[i+1][j-1]+2*img1[i+1][j]+img1[i+1][j+1]-img1[i-1][j-1]-2*img1[i-1][j]-img1[i-1][j+1]), 2)+np.power((img1[i-1][j+1]+2*img1[i][j+1]+img1[i+1][j+1]-img1[i-1][j-1]-2*img1[i][j-1]-img1[i+1][j-1]), 2))
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def Frei_Chen(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(1,h-1):
        for j in range(1,w-1):
            result[i][j] = np.sqrt(np.power((img1[i+1][j-1]+np.sqrt(2)*img1[i+1][j]+img1[i+1][j+1]-img1[i-1][j-1]-np.sqrt(2)*img1[i-1][j]-img1[i-1][j+1]), 2)+np.power((img1[i-1][j+1]+np.sqrt(2)*img1[i][j+1]+img1[i+1][j+1]-img1[i-1][j-1]-np.sqrt(2)*img1[i][j-1]-img1[i+1][j-1]), 2))
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def Kirsch(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(1,h-1):
        for j in range(1,w-1):
            k0 = 5*(img1[i-1][j+1]+img1[i][j+1]+img1[i+1][j+1])-3*(img1[i-1][j]+img1[i+1][j]+img1[i-1][j-1]+img1[i][j-1]+img1[i+1][j-1])
            k1 = 5*(img1[i-1][j+1]+img1[i][j+1]+img1[i-1][j])-3*(img1[i+1][j+1]+img1[i+1][j]+img1[i-1][j-1]+img1[i][j-1]+img1[i+1][j-1])
            k2 = 5*(img1[i-1][j+1]+img1[i-1][j-1]+img1[i-1][j])-3*(img1[i+1][j+1]+img1[i+1][j]+img1[i][j+1]+img1[i][j-1]+img1[i+1][j-1])
            k3 = 5*(img1[i][j-1]+img1[i-1][j-1]+img1[i-1][j])-3*(img1[i+1][j+1]+img1[i+1][j]+img1[i][j+1]+img1[i-1][j+1]+img1[i+1][j-1])
            k4 = 5*(img1[i][j-1]+img1[i-1][j-1]+img1[i+1][j-1])-3*(img1[i+1][j+1]+img1[i+1][j]+img1[i][j+1]+img1[i-1][j+1]+img1[i-1][j])
            k5 = 5*(img1[i][j-1]+img1[i+1][j]+img1[i+1][j-1])-3*(img1[i+1][j+1]+img1[i-1][j-1]+img1[i][j+1]+img1[i-1][j+1]+img1[i-1][j])
            k6 = 5*(img1[i+1][j+1]+img1[i+1][j]+img1[i+1][j-1])-3*(img1[i][j-1]+img1[i-1][j-1]+img1[i][j+1]+img1[i-1][j+1]+img1[i-1][j])
            k7 = 5*(img1[i+1][j+1]+img1[i+1][j]+img1[i][j+1])-3*(img1[i][j-1]+img1[i-1][j-1]+img1[i+1][j-1]+img1[i-1][j+1]+img1[i-1][j])
            result[i][j] = max(k0, k1, k2, k3, k4, k5, k6, k7)
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result


def Robinson(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(1,h-1):
        for j in range(1,w-1):
            r0 = (-1)*img1[i-1][j-1]+(-2)*img1[i][j-1]+(-1)*img1[i+1][j-1]+0*img1[i+1][j]+img1[i+1][j+1]+2*img1[i][j+1]+img1[i-1][j+1]+0*img1[i-1][j]
            r1 = 0*img1[i-1][j-1]+(-1)*img1[i][j-1]+(-2)*img1[i+1][j-1]+(-1)*img1[i+1][j]+0*img1[i+1][j+1]+img1[i][j+1]+2*img1[i-1][j+1]+img1[i-1][j]
            r2 = img1[i-1][j-1]+0*img1[i][j-1]+(-1)*img1[i+1][j-1]+(-2)*img1[i+1][j]+(-1)*img1[i+1][j+1]+0*img1[i][j+1]+img1[i-1][j+1]+2*img1[i-1][j]
            r3 = 2*img1[i-1][j-1]+img1[i][j-1]+0*img1[i+1][j-1]+(-1)*img1[i+1][j]+(-2)*img1[i+1][j+1]+(-1)*img1[i][j+1]+0*img1[i-1][j+1]+img1[i-1][j]
            r4 = img1[i-1][j-1]+2*img1[i][j-1]+img1[i+1][j-1]+0*img1[i+1][j]+(-1)*img1[i+1][j+1]+(-2)*img1[i][j+1]+(-1)*img1[i-1][j+1]+0*img1[i-1][j]
            r5 = 0*img1[i-1][j-1]+img1[i][j-1]+2*img1[i+1][j-1]+img1[i+1][j]+0*img1[i+1][j+1]+(-1)*img1[i][j+1]+(-2)*img1[i-1][j+1]+(-1)*img1[i-1][j]
            r6 = (-1)*img1[i-1][j-1]+0*img1[i][j-1]+img1[i+1][j-1]+2*img1[i+1][j]+img1[i+1][j+1]+0*img1[i][j+1]+(-1)*img1[i-1][j+1]+(-2)*img1[i-1][j]
            r7 = (-2)*img1[i-1][j-1]+(-1)*img1[i][j-1]+0*img1[i+1][j-1]+img1[i+1][j]+2*img1[i+1][j+1]+img1[i][j+1]+0*img1[i-1][j+1]+(-1)*img1[i-1][j]
            result[i][j] = max(r0, r1, r2, r3, r4, r5, r6, r7)
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def Nevatia(original_img, threshold):
    h, w, c = original_img.shape
    img1 = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            img1[i][j] = original_img[i][j][0] 
    result = np.zeros([h,w])
       
    for i in range(2,h-2):
        for j in range(2,w-2):
            # N0~N5 分別是 0,30,60,-90,-60,-30 度
            N0 = 100*(img1[i-2][j-2]+img1[i-2][j-1]+img1[i-2][j]+img1[i-2][j+1]+img1[i-2][j+2]+img1[i-1][j-2]+img1[i-1][j-1]+img1[i-1][j]+img1[i-1][j+1]+img1[i-1][j+2])+0*(img1[i][j-2]+img1[i][j-1]+img1[i][j]+img1[i][j+1]+img1[i][j+2])+(-100)*(img1[i+1][j-2]+img1[i+1][j-1]+img1[i+1][j]+img1[i+1][j+1]+img1[i+1][j+2]+img1[i+2][j-2]+img1[i+2][j-1]+img1[i+2][j]+img1[i+2][j+1]+img1[i+2][j+2])
            N1 = 100*(img1[i-2][j-2]+img1[i-2][j-1]+img1[i-2][j]+img1[i-2][j+1]+img1[i-2][j+2]+img1[i-1][j-2]+img1[i-1][j-1]+img1[i-1][j])+78*img1[i-1][j+1]+(-32)*img1[i-1][j+2]+100*img1[i][j-2]+92*img1[i][j-1]+0*img1[i][j]+(-92)*(img1[i][j+1])+(-100)*img1[i][j+2]+32*img1[i+1][j-2]+(-78)*img1[i+1][j-1]+(-100)*(img1[i+1][j]+img1[i+1][j+1]+img1[i+1][j+2]+img1[i+2][j-2]+img1[i+2][j-1]+img1[i+2][j]+img1[i+2][j+1]+img1[i+2][j+2])
            N2 = 100*(img1[i-2][j-2]+img1[i-2][j-1]+img1[i-2][j])+32*img1[i-2][j+1]+(-100)*img1[i-2][j+2]+100*(img1[i-1][j-2]+img1[i-1][j-1])+92*img1[i-1][j]+(-78)*img1[i-1][j+1]+(-100)*img1[i-1][j+2]+100*(img1[i][j-2]+img1[i][j-1])+0*img1[i][j]+(-100)*(img1[i][j+1]+img1[i][j+2])+100*img1[i+1][j-2]+78*img1[i+1][j-1]+(-92)*img1[i+1][j]+(-100)*(img1[i+1][j+1]+img1[i+1][j+2])+100*img1[i+2][j-2]+(-32)*img1[i+2][j-1]+(-100)*(img1[i+2][j]+img1[i+2][j+1]+img1[i+2][j+2])
            N3 = (-100)*(img1[i-2][j-2]+img1[i-2][j-1])+0*img1[i-2][j]+100*(img1[i-2][j+1]+img1[i-2][j+2])+(-100)*(img1[i-1][j-2]+img1[i-1][j-1])+0*img1[i-1][j]+100*(img1[i-1][j+1]+img1[i-1][j+2])+(-100)*(img1[i][j-2]+img1[i][j-1])+0*img1[i][j]+100*(img1[i][j+1]+img1[i][j+2])+(-100)*(img1[i+1][j-2]+img1[i+1][j-1])+0*img1[i+1][j]+100*(img1[i+1][j+1]+img1[i+1][j+2])+(-100)*(img1[i+2][j-2]+img1[i+2][j-1])+0*img1[i+2][j]+100*(img1[i+2][j+1]+img1[i+2][j+2])
            N4 = (-100)*img1[i-2][j-2]+32*img1[i-2][j-1]+100*(img1[i-2][j]+img1[i-2][j+1]+img1[i-2][j+2])+(-100)*img1[i-1][j-2]+(-78)*img1[i-1][j-1]+92*img1[i-1][j]+100*(img1[i-1][j+1]+img1[i-1][j+2])+(-100)*(img1[i][j-2]+img1[i][j-1])+0*img1[i][j]+100*(img1[i][j+1]+img1[i][j+2])+(-100)*(img1[i+1][j-2]+img1[i+1][j-1])+(-92)*img1[i+1][j]+78*img1[i+1][j+1]+100*img1[i+1][j+2]+(-100)*(img1[i+2][j-2]+img1[i+2][j-1]+img1[i+2][j])+(-32)*img1[i+2][j+1]+100*img1[i+2][j+2]
            N5 = 100*(img1[i-2][j-2]+img1[i-2][j-1]+img1[i-2][j]+img1[i-2][j+1]+img1[i-2][j+2])+(-32)*img1[i-1][j-2]+78*img1[i-1][j-1]+100*(img1[i-1][j]+img1[i-1][j+1]+img1[i-1][j+2])+(-100)*img1[i][j-2]+(-92)*img1[i][j-1]+0*img1[i][j]+92*img1[i][j+1]+100*img1[i][j+2]+(-100)*(img1[i+1][j-2]+img1[i+1][j-1]+img1[i+1][j])+(-78)*img1[i+1][j+1]+32*img1[i+1][j+2]+(-100)*(img1[i+2][j-2]+img1[i+2][j-1]+img1[i+2][j]+img1[i+2][j+1]+img1[i+2][j+2])
            result[i][j] = max(N0, N1, N2, N3, N4, N5)
    
    for i in range(h):
        for j in range(w):  
            if result[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

if __name__ == '__main__':
    original_img = cv2.imread("lena.bmp")
    
    Roberts_img = Roberts(original_img, 12)
    cv2.imwrite("Roberts.jpg", Roberts_img)

    Prewitt_img = Prewitt(original_img, 24)
    cv2.imwrite("Prewitt.jpg", Prewitt_img)

    Sobel_img = Sobel(original_img, 38)
    cv2.imwrite("Sobel.jpg", Sobel_img)

    Frei_Chen_img = Frei_Chen(original_img, 30)
    cv2.imwrite("Frei_Chen.jpg", Frei_Chen_img)

    Kirsch_img = Kirsch(original_img, 135)
    cv2.imwrite("Kirsch.jpg", Kirsch_img)

    Robinson_img = Robinson(original_img, 43)
    cv2.imwrite("Robinson.jpg", Robinson_img)

    Nevatia_img = Nevatia(original_img, 12500)
    cv2.imwrite("Nevatia.jpg", Nevatia_img)

