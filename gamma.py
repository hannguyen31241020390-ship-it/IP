import cv2
import numpy as np
image= cv2.imread("C:/Users/Windows/Downloads/Fig0308(a)(fractured_spine).png", cv2.IMREAD_GRAYSCALE)
image1= cv2.imread("C:/Users/Windows/Downloads/Fig0309(a)(washed_out_aerial_image).png", cv2.IMREAD_GRAYSCALE)
gamma=1.5
c=(256-1)**(1-gamma)
img_gam=np.uint8(c*image**gamma)
img_gam1=np.uint8(c*image1**gamma)
cv2.imshow("Original", image)
cv2.imshow("Gamma Corrected", img_gam)
cv2.imshow("Original 1", image1)
cv2.imshow("Gamma Corrected 1", img_gam1)

cv2.waitKey(0)  
cv2.destroyAllWindows()
#cv2.imwrite('C:/Users/Windows/Downloads/imageskhac.jpg', image)