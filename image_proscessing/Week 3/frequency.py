import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_fft_image(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def low_pass_filter(img, percent):
    fshift = get_fft_image(img)
    
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    mask = np.zeros([rows, cols], dtype=np.uint8)
    mask[int(midr-percent*rows//2):int(midr+percent*rows//2), int(midc-percent*cols//2):int(midc+percent*cols//2)] = 1
    
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    
    return np.abs(out)

def high_pass_filter(img, percent):
    fshift = get_fft_image(img)
    
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    fshift[int(midr-percent*rows//2):int(midr+percent*rows//2), int(midc-percent*cols//2):int(midc+percent*cols//2)] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    
    return np.abs(out)

if __name__ == "__main__":
    img = cv2.imread("image/Lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = low_pass_filter(gray, 0.1)
    edge = high_pass_filter(gray, 0.1)

    plt.subplot(131), plt.imshow(gray, cmap = 'gray')
    plt.title("Origin")
    plt.subplot(132); plt.imshow(blur, cmap="gray")
    plt.title("Low pass filter")
    plt.subplot(133); plt.imshow(edge, cmap="gray")
    plt.title("High pass filter")
    plt.show()
