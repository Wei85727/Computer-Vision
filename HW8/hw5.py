from PIL import Image

kernel = [[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,0],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-1],[2,0],[2,1]]

image_original = Image.open("lena.bmp")

def dilation(image_original, kernel):
    image_dilation = Image.new('L', image_original.size)
    for i in range(image_original.size[0]):
        for j in range(image_original.size[1]):
            temp_Max = 0
            for element in kernel:
                x, y = element
                if (0 <= i+x <= image_original.size[0]-1) and (0 <= j+y <= image_original.size[0]-1):
                    pixelvalue = image_original.getpixel((i+x, j+y))
                    temp_Max = max(temp_Max, pixelvalue)
            image_dilation.putpixel((i,j),temp_Max)  
    return image_dilation              

def erosion(image_original, kernel):
    image_erosion = Image.new('L', image_original.size)
    for i in range(image_original.size[0]):
        for j in range(image_original.size[1]):
            temp_min = 255
            for element in kernel:
                x, y = element
                if (0 <= i+x <= image_original.size[0]-1) and (0 <= j+y <= image_original.size[0]-1):
                    pixelvalue = image_original.getpixel((i+x, j+y))
                    temp_min = min(temp_min, pixelvalue)
            image_erosion.putpixel((i,j),temp_min)  
    return image_erosion

def main():
    res_dil = dilation(image_original, kernel)
    res_dil.save("dilation.bmp")
    
    res_ero = erosion(image_original, kernel)
    res_ero.save("erosion.bmp")

    res_ope = dilation(erosion(image_original, kernel), kernel)
    res_ope.save("opening.bmp")

    res_clo = erosion(dilation(image_original, kernel), kernel)
    res_clo.save("closing.bmp")

if __name__ == '__main__':
    main()        