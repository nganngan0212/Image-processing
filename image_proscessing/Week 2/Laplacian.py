import cv2
import numpy as np
# import math

def laplacian(img):
    ker_mat = np.array([[1,-2, 1],[-2, 4, -2],[1, -2, 1]]) 

    rows, cols = img.shape

    ex_mat = np.zeros((rows+2, cols+2), dtype=np.uint8)
    ex_mat[1:1+rows, 1:1+cols] = img

    
    out = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            local = ex_mat[i:i+3,j:j+3]
            out[i,j] = np.sum(ker_mat*local)

    out = np.sqrt(out**2)
    out = out.astype('uint8') 
    return out

input = cv2.imread('image/Lenna.png', 0)
cv2.imshow('original', input)

output = laplacian(input)

cv2.imshow('Output image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
