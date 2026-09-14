import cv2
import numpy as np

# 1. Đọc ảnh màu BGR
img = cv2.imread("C:/Users/Windows/Downloads/Fig0630(01)(strawberries_fullcolor).png")

# 2. Chuyển sang không gian màu HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Định nghĩa dải màu đỏ trong không gian HSV
# Trong OpenCV: H chạy từ 0-179. Màu đỏ nằm ở 2 đầu dải: [0, 10] và [170, 180]
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
# 4. Tạo mặt nạ nhị phân (Mask)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 5. Cắt lát màu: Áp mặt nạ lên ảnh gốc (vùng ngoài biến thành đen)
result = cv2.bitwise_and(img, img, mask=mask)

# 6. Hiển thị
cv2.imshow("Original Image", img)
cv2.imshow("Color Slicing (Red Strawberries)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()