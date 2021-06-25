import cv2
import numpy as np
# import math

def sobel(img):
    ker_mat_x = np.array([[-1,-2, -1],[0, 0, 0],[1, 2, 1]]) 
    ker_mat_y = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) 

    rows, cols = img.shape

    ex_mat = np.zeros((rows+2, cols+2), dtype=np.uint8)
    ex_mat[1:1+rows, 1:1+cols] = img

    # ex_mat[1:1+rows, 0] = img[0:rows, 0]
    # ex_mat[1:1+rows, cols+1] = img[0:rows, cols-1]
    # ex_mat[0, 1:1+cols] = img[0, 0:cols]
    # ex_mat[rows+1, 1:1+cols] = img[rows-1, 0:cols]

    out_x = np.zeros((rows, cols))
    out_y = np.zeros((rows, cols))
    out = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            local = ex_mat[i:i+3,j:j+3]
            out_x[i,j] = np.sum(ker_mat_x*local)
            out_y[i,j] = np.sum(ker_mat_y*local)
    
    out = np.sqrt(out_x**2 + out_y**2)
    out = out.astype('uint8') 
    return out

input = cv2.imread('image/Lenna.png', 0)
cv2.imshow('original', input)

output = sobel(input)

cv2.imshow('Output image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
