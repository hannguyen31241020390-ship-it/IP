import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BƯỚC (a): ĐỌC ẢNH GỐC
# ==========================================
img_a = cv2.imread("C:/Users/Windows/Downloads/Fig0343(a)(skeleton_orig).png", cv2.IMREAD_GRAYSCALE)

# Chuyển sang float64 để tính toán ma trận không bị tràn số
f = img_a.astype(np.float64)

# ==========================================
# 2. BƯỚC (b): BỘ LỌC LAPLACIAN CỦA (a)
# ==========================================
kernel_laplacian = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float64)
laplacian_raw = cv2.filter2D(f, -1, kernel_laplacian)

# Chuẩn hóa về [0, 255] để hiển thị nền xám trung tính
img_b = cv2.normalize(laplacian_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# ==========================================
# 3. BƯỚC (c): LÀM NÉT ẢNH BẰNG LAPLACIAN
# ==========================================
# Khi dùng kernel có hệ số tâm âm: Sharpened = Original - Laplacian_raw
img_c_raw = f - laplacian_raw
img_c = np.clip(img_c_raw, 0, 255).astype(np.uint8)

# ==========================================
# 4. BƯỚC (d): GRADIENT SOBEL CỦA (a)
# ==========================================
sobel_x = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
img_d = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# ==========================================
# 5. BƯỚC (e): LÀM MƯỢT SOBEL BẰNG BỘ LỌC 5x5
# ==========================================
kernel_avg = np.ones((5, 5), dtype=np.float64) / 25.0
img_e_raw = cv2.filter2D(sobel_mag, -1, kernel_avg)
img_e = cv2.normalize(img_e_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# ==========================================
# 6. BƯỚC (f): TẠO MẶT NẠ MASK = (c) * (e)
# ==========================================
# Chuẩn hóa về dải [0, 1] trước khi nhân
c_norm = img_c_raw / 255.0
e_norm = cv2.normalize(img_e_raw, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
mask_f_raw = (c_norm * e_norm) * 255.0
img_f = np.clip(mask_f_raw, 0, 255).astype(np.uint8)

# ==========================================
# 7. BƯỚC (g): CỘNG MẶT NẠ VÀO ẢNH GỐC = (a) + (f)
# ==========================================
img_g_raw = f + mask_f_raw
img_g = np.clip(img_g_raw, 0, 255).astype(np.uint8)

# ==========================================
# 8. BƯỚC (h): BIẾN ĐỔI POWER-LAW (GAMMA = 0.5)
# ==========================================
gamma = 0.5
c_const = 1.0
g_norm = img_g_raw / 255.0
img_h_raw = c_const * (g_norm ** gamma) * 255.0
img_h = np.clip(img_h_raw, 0, 255).astype(np.uint8)

# ==========================================
# HIỂN THỊ TỔNG HỢP CẢ 8 ẢNH TRÊN 2 HÀNG
# ==========================================
titles = [
    "(a) Bone scan image",
    "(b) Laplacian of (a)",
    "(c) Sharpened: (a) - (b)",
    "(d) Sobel gradient of (a)",
    "(e) Smoothed Sobel (5x5)",
    "(f) Mask: (c) * (e)",
    "(g) Sharpened: (a) + (f)",
    "(h) Power-law on (g)"
]

images = [img_a, img_b, img_c, img_d, img_e, img_f, img_g, img_h]

plt.figure(figsize=(18, 9))
for idx in range(8):
    plt.subplot(2, 4, idx + 1)
    plt.imshow(images[idx], cmap="gray")
    plt.title(titles[idx], fontsize=11, pad=8)
    plt.axis("off")

plt.tight_layout()
plt.show()