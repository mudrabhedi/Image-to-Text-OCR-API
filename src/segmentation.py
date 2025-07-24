import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image in grayscale
image = cv2.imread(r'C:\DIP PROJECT\dip mini project\images\image9.jpg', 0)

# Apply global thresholding as a baseline
_, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding with different configurations
adaptive_thresh1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
adaptive_thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 15, 4)
adaptive_thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
adaptive_thresh4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 4)

# Display the original and thresholded images
plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(2, 3, 2), plt.imshow(binary_thresh, cmap='gray'), plt.title("Global Thresholding")

# Display the adaptive thresholding results
plt.subplot(2, 3, 3), plt.imshow(adaptive_thresh1, cmap='gray')
plt.title("Adaptive MEAN_C, Block=11, C=2")

plt.subplot(2, 3, 4), plt.imshow(adaptive_thresh2, cmap='gray')
plt.title("Adaptive MEAN_C, Block=15, C=4")

plt.subplot(2, 3, 5), plt.imshow(adaptive_thresh3, cmap='gray')
plt.title("Adaptive GAUSSIAN_C, Block=11, C=2")

plt.subplot(2, 3, 6), plt.imshow(adaptive_thresh4, cmap='gray')
plt.title("Adaptive GAUSSIAN_C, Block=15, C=4")

plt.tight_layout()
plt.show()
