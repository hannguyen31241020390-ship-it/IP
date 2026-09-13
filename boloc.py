import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread("C:/Users/Windows/Downloads/Fig0333(a)(test_pattern_blurring_orig).png")

# Chuyển BGR sang RGB để matplotlib hiển thị đúng màu
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kernel làm mờ 5x5
kernel = np.ones((15,15), dtype=np.float32) / (15*15)

# Áp dụng bộ lọc kernel
result = cv2.filter2D(image_rgb, -1, kernel)

# Hiển thị ảnh gốc và ảnh sau lọc
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Ảnh sau kernel 15x15")
plt.axis("off")

plt.tight_layout()
plt.show()

# Lưu kết quả
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_kernel_15x15.jpg", result_bgr)

cv2.waitKey(0)  
cv2.destroyAllWindows()