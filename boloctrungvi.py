import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread("C:/Users/Windows/Downloads/Fig0335(a)(ckt_board_saltpep_prob_pt05).png")

# Chuyển BGR sang RGB để matplotlib hiển thị đúng màu
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kernel làm mờ 5x5
kernel = np.ones((5,5), dtype=np.float32) / 25

# Áp dụng bộ lọc kernel
# Áp dụng bộ lọc trung bình và trung vị
mean_result = cv2.filter2D(image_rgb, -1, kernel)
median_result = cv2.medianBlur(image_rgb, 5)

# Hiển thị ảnh gốc và ảnh sau lọc
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mean_result)
plt.title("Lọc trung bình 9x9")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(median_result)
plt.title("Lọc trung vị 9x9")
plt.axis("off")

plt.tight_layout()
plt.show()

# Lưu kết quả
mean_result_bgr = cv2.cvtColor(mean_result, cv2.COLOR_RGB2BGR)
median_result_bgr = cv2.cvtColor(median_result, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_mean_9x9.jpg", mean_result_bgr)
cv2.imwrite("output_median_9x9.jpg", median_result_bgr)

cv2.waitKey(0)  
cv2.destroyAllWindows()