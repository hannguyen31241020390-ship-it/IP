import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
image= cv2.imread("C:/Users/Windows/Downloads/Fig0310(b)(washed_out_pollen_image).png", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 4))
plt.hist(image.ravel(), 256, [0, 256])
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Biểu đồ Histogram của ảnh xám")
plt.show()