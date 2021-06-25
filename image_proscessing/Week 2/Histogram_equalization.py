import numpy as np
import cv2

def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        Avg = (R)
        grayImage = R

        return grayImage       

def frequency_of_pixels(img):
    rows,cols = img.shape

    arr = np.zeros((255,), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            val = img[i][j]
            arr[val] = arr[val]+1
    return arr

image = cv2.imread("lenna.jpg")
cv2.imshow('Original', image)    

rows, cols = image.shape[:2]

grayImage = rgb_to_gray(image)  

freq_val = frequency_of_pixels(grayImage)
# print("Frequency of pixels value 0->255",arr)
prob_val = np.arr((255,))
for i in range(255):
    prob_val[i] = freq_val[i]/(rows*cols)


cv2.imshow('RGB', grayImage) 
cv2.waitKey(0)  
cv2.destroyAllWindows()  
