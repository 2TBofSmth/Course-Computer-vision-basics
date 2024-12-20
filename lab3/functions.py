import numpy as np


def to_grayscale(img):
    height, width, _ = img.shape
    grayscale_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = img[i, j]
            gray_value = 0.11 * pixel[0] + 0.53 * pixel[1] + 0.36 * pixel[2]
            grayscale_img[i, j] = gray_value

    return grayscale_img


def img_filter(img, kernel, k):
    image_h, image_w = img.shape
    kernel_h, kernel_w = kernel.shape

    output_h, output_w = image_h - kernel_h + 1, image_w - kernel_w + 1

    output_img = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = img[i:i + kernel_h, j:j + kernel_w]
            output_img[i, j] = np.sum(region * kernel * k)

    return output_img


def shift_img(image):
    height, width = image.shape

    shifted_img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            x1 = x + 10
            y1 = y + 20

            if 0 <= x1 < width and 0 <= y1 < height:
                shifted_img[y1, x1] = image[y, x]

    return shifted_img


inversion_kernel = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, -1, -1, -1, -1, -1, 0],
    [0, -1, -1, -1, -1, -1, 0],
    [0, -1, -1, -1, -1, -1, 0],
    [0, -1, -1, -1, -1, -1, 0],
    [0, -1, -1, -1, -1, -1, 0],
    [0, 0, 0, 0, 0, 0, 0]])

gaussian_kernel = np.array([
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2],
    [2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2],
    [2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2],
    [2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]])

diagonal_movement_blur_kernel = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]])

sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]])

sobel_kernel = np.array([
    [ -1, -1, -2, -2, -1, -1],
    [ -1, -1, -2, -2, -1, -1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 2, 2, 1, 1],
    [1, 1, 2, 2, 1, 1]])

border_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

almost_my_kernel = np.array([
    [0,  0,  1,  0,  0],
    [0,  1,  2,  1,  0],
    [1,  2,  4,  2,  1],
    [0,  1,  2,  1,  0],
    [0,  0,  1,  0,  0]
])
my_kernel = almost_my_kernel / np.sum(almost_my_kernel)