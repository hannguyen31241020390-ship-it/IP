import cv2
import matplotlib.pyplot as plt

# 1. Đọc ảnh vệ tinh / ảnh đo đạc ở dạng ảnh xám (Grayscale)
img_gray = cv2.imread("C:/Users/Windows/Downloads/54e852d5-df26-4b83-b27f-ca3fdcfb4d0f.png", cv2.IMREAD_GRAYSCALE)
# 2. Áp dụng bảng màu giả (Pseudocolor / False Color)
# Dùng COLORMAP_PARULA như trên slide (có từ OpenCV 3.3+)
# Ngoài ra có thể thử: COLORMAP_JET, COLORMAP_RAINBOW, COLORMAP_VIRIDIS
img_pseudo = cv2.applyColorMap(img_gray, cv2.COLORMAP_VIRIDIS)  # Thay đổi COLORMAP nếu muốn thử các bảng màu khác

# Chuyển từ định dạng màu BGR mặc định của OpenCV sang RGB để hiển thị đúng bằng Matplotlib
img_pseudo_rgb = cv2.cvtColor(img_pseudo, cv2.COLOR_BGR2RGB)

# 3. Hiển thị đối chiếu ảnh đơn sắc gốc và ảnh màu giả
plt.figure(figsize=(12, 5))

# Ảnh xám gốc
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Original Monochrome Image")
plt.axis("off")

# Ảnh Pseudocolor
plt.subplot(1, 2, 2)
plt.imshow(img_pseudo_rgb)
plt.title("Pseudocolor Image (COLORMAP_PARULA)")
plt.axis("off")

plt.tight_layout()
plt.show()