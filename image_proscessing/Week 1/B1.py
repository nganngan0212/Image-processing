import cv2
import numpy as np

input = cv2.imread('image\salt_peper.jpg', 0)
cv2.imshow('original', input)

#use Lenna.png image for task inverse image
def inverse_image(img):
    rows, cols = img.shape
    mask = np.ones((rows, cols), dtype=np.uint8)*255
    out = mask - img
    return out

#use salt_peper.jpg for 2 tasks meadian filter and mean filter

def mean_filter(img, ker=3):
    ker_mat = np.ones((ker, ker))/(ker**2)

    rows, cols = img.shape

    ex_mat = np.zeros((rows+ker-1, cols+ker-1), dtype=np.uint8)
    ex_mat[ker//2:ker//2+rows, ker//2:ker//2+cols] = img

    ex_mat[ker//2:ker//2+rows, 0] = img[0:rows, 0]
    ex_mat[ker//2:ker//2+rows, cols+ker//2] = img[0:rows, cols-1]
    ex_mat[0, ker//2:ker//2+cols] = img[0, 0:cols]
    ex_mat[rows+ker//2, ker//2:ker//2+cols] = img[rows-1, 0:cols]

    out = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            local = ex_mat[i:i+ker, j:j+ker]
            out[i,j] = np.sum(ker_mat*local)
    return out

def median_filter(img, ker=3):
    rows, cols = img.shape

    ex_mat = np.zeros((rows+ker-1, cols+ker-1), dtype=np.uint8)
    ex_mat[ker//2:ker//2+rows, ker//2:ker//2+cols] = img

    ex_mat[ker//2:ker//2+rows, 0] = img[0:rows, 0]
    ex_mat[ker//2:ker//2+rows, cols+ker//2] = img[0:rows, cols-1]
    ex_mat[0, ker//2:ker//2+cols] = img[0, 0:cols]
    ex_mat[rows+ker//2, ker//2:ker//2+cols] = img[rows-1, 0:cols]

    out = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            local = ex_mat[i:i+ker, j:j+ker]
            arr = np.reshape(local, ker*ker)
            arr = np.sort(arr)
            ker_value = arr[ker+ker//2]
            out[i,j] = ker_value
    return out


# change the function
output = median_filter(input)

cv2.imshow('Output image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

