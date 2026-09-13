import cv2
import numpy as np
image= cv2.imread("C:/Users/Windows/Downloads/Fig0310(b)(washed_out_pollen_image).png", cv2.IMREAD_GRAYSCALE)

rmin= np.min(image)
rmax= np.max(image)
rmean= np.mean(image)

def ConstrastStretching(image, r1, r2, s1, s2):
    W, H = image.shape
    img_constrast= np.zeros((W,H),np.uint8)
    for x in range(0,W):
        for y in range(0,H):
            r= image[x,y]
            if r<=r1:
                s= (s1/r1)*r
            elif ((r >r1)& (r<=r2)):
                s= (s2-s1)/(r2-r1)*r + (s1*r2-s2*r1)/(r2-r1)
            else:
                s= (255-s2)/(255-r2)*r + (255*(s2-r2))/(255-r2)
            img_constrast[x,y]= np.uint8(s)
    return img_constrast

img_constrast=ConstrastStretching(image, rmin, rmax, 0.0, 255.0)
img_threshold= ConstrastStretching(image, rmean, rmean, 0.0, 255.0)
cv2.imshow("Original", image)
cv2.imshow("Contrast Stretched", img_constrast)
cv2.imshow("Threshold", img_threshold)

cv2.waitKey(0)  
cv2.destroyAllWindows()