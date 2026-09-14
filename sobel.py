import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh thang độ xám
img = cv2.imread("C:/Users/Windows/Downloads/Fig0342(a)(contact_lens_original).png", cv2.IMREAD_GRAYSCALE)
# Chuyển sang float32 để tính gradient không bị tràn số
f = img.astype(np.float32)

# 2. Tính đạo hàm Sobel theo 2 trục X và Y
# ksize=3 tương ứng kernel 3x3 như trong slide lý thuyết
gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)

# 3. Tính độ lớn Gradient M(x, y) = sqrt(gx^2 + gy^2)
sobel_mag = cv2.magnitude(gx, gy)

# 4. Ép dải về [0, 255] và chuyển về uint8 để hiển thị
sobel_display = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# 5. Hiển thị đối chiếu 2 ảnh
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("(a) Optical image of contact lens")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sobel_display, cmap="gray")
plt.title("(b) Sobel gradient")
plt.axis("off")

plt.tight_layout()
plt.show()