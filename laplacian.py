import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh gốc ở dạng grayscale (Ảnh a)
img = cv2.imread("C:/Users/Windows/Downloads/Fig0338(a)(blurry_moon).png", cv2.IMREAD_GRAYSCALE)

f = img.astype(np.float64)

# 2. Định nghĩa các kernel Laplacian có hệ số tâm âm
# Kernel 4 lân cận (cho b, c, d)
kernel_4 = np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]], dtype=np.float64)

# Kernel mở rộng 8 lân cận (cho e)
kernel_8 = np.array([[1,  1, 1],
                     [1, -8, 1],
                     [1,  1, 1]], dtype=np.float64)

# 3. Tính toán các ảnh thành phần
# Lọc với kernel 4 lân cận
laplacian_4 = cv2.filter2D(f, -1, kernel_4)

# (b) Laplacian không co giãn tỷ lệ (cắt ngưỡng về [0, 255] trực tiếp)
img_b = np.clip(laplacian_4, 0, 255).astype(np.uint8)

# (c) Laplacian có co giãn tỷ lệ (scale toàn bộ dải giá trị về [0, 255])
img_c = cv2.normalize(laplacian_4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
img_c = img_c.astype(np.uint8)

# (d) Làm nét bằng kernel 4 lân cận: g = f - laplacian (do tâm kernel âm)
img_d = f - laplacian_4
img_d = np.clip(img_d, 0, 255).astype(np.uint8)

# (e) Làm nét bằng kernel mở rộng 8 lân cận
laplacian_8 = cv2.filter2D(f, -1, kernel_8)
img_e = f - laplacian_8
img_e = np.clip(img_e, 0, 255).astype(np.uint8)

# 4. Hiển thị kết quả
titles = [
    '(a) Blurred image',
    '(b) Laplacian without scaling',
    '(c) Laplacian with scaling',
    '(d) Sharpened (4-neighbor)',
    '(e) Sharpened (8-neighbor extension)'
]
images = [img, img_b, img_c, img_d, img_e]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()