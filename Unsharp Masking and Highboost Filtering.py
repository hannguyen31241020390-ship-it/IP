import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh thang độ xám
img = cv2.imread("C:/Users/Windows/Downloads/Fig0340(a)(dipxe_text).png", cv2.IMREAD_GRAYSCALE)
# 2. Chuyển sang float32 để tính toán không bị tràn số
img_f = img.astype(np.float32)

# 3. Tạo bộ lọc trung bình làm mờ ảnh (k = 5)
k = 5
kernel = np.ones((k, k), np.float32) / (k * k)
img_blur = cv2.filter2D(img_f, -1, kernel)

# 4. Tạo mặt nạ Unsharp Mask (ảnh gốc - ảnh mờ)
mask = img_f - img_blur

# 5. Làm nét ảnh
# k_sharp = 1.0: Unsharp Masking
# k_sharp > 1.0: Highboost Filtering (ví dụ thử k_sharp = 2.0 để tăng biên mạnh hơn)
k_sharp = 1.0
sharp = img_f + k_sharp * mask

# 6. Ép dải giá trị về [0, 255] và chuyển về uint8 để hiển thị
sharp_display = np.clip(sharp, 0, 255).astype(np.uint8)
img_blur_display = np.clip(img_blur, 0, 255).astype(np.uint8)

# Chuẩn hóa riêng ma trận mask để quan sát được các đường biên trên nền xám
mask_display = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 7. Hiển thị so sánh
titles = ["Original", "Blurred (k=5)", "Unsharp Mask", f"Sharpened (k={k_sharp})"]
images = [img, img_blur_display, mask_display, sharp_display]

plt.figure(figsize=(14, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()