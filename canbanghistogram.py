import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
image= cv2.imread("C:/Users/Windows/Downloads/Fig0310(b)(washed_out_pollen_image).png", cv2.IMREAD_GRAYSCALE)
def HistogramCalculate(img):
    W, H = img.shape
    hr = np.zeros(256)
    for x in range(W):
        for y in range(H):
            val = img[x, y]
            hr[val] += 1
    return hr
def HistogramEqualization(image):
    W, H= image.shape
    hr= HistogramCalculate(image)
    pr = np.zeros(256)
    for i in range(256):
        pr[i] = hr[i]/(W*H)
    hr_eq = np.zeros(256)
    for i in range(256):
        for j in range(i+1):
            hr_eq[i] += pr[j]
    img_eq = np.zeros((W, H), dtype=np.uint8)
    for i in range(W):
        for j in range(H):
            r= image[i][j]
            img_eq[i][j] = hr_eq[r]*255
    return img_eq
cv2.imshow("Original Image", image)
cv2.imshow("Histogram Equalization", HistogramEqualization(image))
cv2.waitKey(0)
cv2.destroyAllWindows()