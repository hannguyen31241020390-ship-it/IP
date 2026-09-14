import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh mức xám
path = "C:/Users/Windows/Downloads/Fig0338(a)(blurry_moon).png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# --- 1. TOÁN TỬ SOBEL ---
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)
sobel_edge = np.clip(sobel_edge, 0, 255).astype(np.uint8)

# --- 2. TOÁN TỬ ROBERTS ---
kernel_rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
kernel_ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(img, cv2.CV_64F, kernel_rx)
roberts_y = cv2.filter2D(img, cv2.CV_64F, kernel_ry)
roberts_edge = cv2.magnitude(roberts_x, roberts_y)
roberts_edge = np.clip(roberts_edge, 0, 255).astype(np.uint8)

# --- HIỂN THỊ SO SÁNH (Ảnh gốc, Roberts, Sobel) ---
plt.figure(figsize=(15, 5))

# 1. Ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Ảnh gốc")
plt.axis("off")

# 2. Toán tử Roberts
plt.subplot(1, 3, 2)
plt.imshow(roberts_edge, cmap='gray')
plt.title("Toán tử Roberts (2x2)")
plt.axis("off")

# 3. Toán tử Sobel
plt.subplot(1, 3, 3)
plt.imshow(sobel_edge, cmap='gray')
plt.title("Toán tử Sobel (3x3)")
plt.axis("off")

plt.tight_layout()
plt.show()