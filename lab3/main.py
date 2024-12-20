from cv2 import imread
import matplotlib.pyplot as plt
from functions import *


image = imread('fff.png')
grayscaled_img = to_grayscale(image)

shifted_img = shift_img(grayscaled_img)
inverted_img = img_filter(grayscaled_img, inversion_kernel, 1)
gaussianed_img = img_filter(grayscaled_img, gaussian_kernel, 1/16)
diagonal_movement_blured_img = img_filter(grayscaled_img, diagonal_movement_blur_kernel, 1/7)
sharpened_img = img_filter(grayscaled_img, sharpening_kernel, 1)
sobeled_img = img_filter(grayscaled_img, sobel_kernel, 1)
bordered_img = img_filter(grayscaled_img, border_kernel, 1)
my_img = img_filter(grayscaled_img, my_kernel, 1)

images = [
    (grayscaled_img, 'Grayscaled Image'),
    (shifted_img, 'Shifted'),
    (inverted_img, 'Inverted'),
    (gaussianed_img, 'Gaussian Blur'),
    (diagonal_movement_blured_img, 'Diagonal Blur'),
    (sharpened_img, 'Sharpened'),
    (sobeled_img, 'Sobel Filter'),
    (bordered_img, 'Border Filter'),
    (my_img, 'My Filter')
]

plt.figure(figsize=(15, 8))

for i, (img, title) in enumerate(images, 1):
    plt.subplot(2, 5, i)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()
