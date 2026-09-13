import cv2
import numpy as np
image= cv2.imread("C:/Users/Windows/Downloads/Fig0312(a)(kidney).png", cv2.IMREAD_GRAYSCALE)

rmean= np.mean(image)
W, H = image.shape
img_slicing1= np.zeros((W,H),np.uint8)
img_slicing2= image.copy()
for x in range(0,W):
    for y in range(0,H):
        r= image[x,y]
        if ((r>=150) & (r<=255)):
            img_slicing1[x,y]= 200
        if ((r>= (rmean-35))& (r<= (rmean+35))):
            img_slicing2[x,y]= 0

cv2.imshow("Original", image)
cv2.imshow("Slicing1", img_slicing1)
cv2.imshow("Slicing2", img_slicing2)

cv2.waitKey(0)  
cv2.destroyAllWindows()