import cv2
image=cv2.imread("C:/Users/Windows/Downloads/Fig0304(a)(breast_digital_Xray).png", cv2.IMREAD_GRAYSCALE)
img_neg= 256-1-image
cv2.imshow("Original Image", image)
cv2.imshow("Negative Image", img_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()
