import cv2
import numpy as np
image=cv2.imread("C:/Users/Windows/Downloads/Fig0305(a)(DFT_no_log).png", cv2.IMREAD_GRAYSCALE)
c= 255/ np.log(255+1)
img_log= np.uint8(c * np.log(image + 1))
cv2.imshow("Original Image", image)
cv2.imshow("Log Transformed Image", img_log)
cv2.waitKey(0)
cv2.destroyAllWindows()
