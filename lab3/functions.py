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


def otsu(grayscale_img):
    height, width = grayscale_img.shape
    histogram = np.zeros(256)
    output_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = grayscale_img[i, j]
            histogram[pixel] += 1

    pixel_amount = height * width
    probs = histogram / pixel_amount

    total_variances = []
    for i in range(256):
        q1 = sum(probs[:i])
        q2 = sum(probs[i + 1:])

        if q1 == 0 or q2 == 0:
            continue

        mu1 = sum([(x * probs[x]) / q1 for x in range(i)])
        mu2 = sum([(x * probs[x]) / q2 for x in range(i + 1, 256)])

        variance_1 = sum([((x - mu1) ** 2) * (probs[x] / q1) for x in range(i)]) ** 2
        variance_2 = sum([((x - mu2) ** 2) * (probs[x] / q2) for x in range(i + 1, 256)]) ** 2

        total_variances.append((q1 * variance_1 + q2 * variance_2) ** 2)
        threshold = total_variances.index(min(total_variances))

        for y in range(height):
            for x in range(width):
                if grayscale_img[y, x] < threshold:
                    output_image[y, x] = 0
                else:
                    output_image[y, x] = 225

    return output_image


def niblek(grayscale_img, obj_color):  # obj_color = w or b
    assert obj_color == 'w' or obj_color == 'b', 'Incorrect obj_color value'
    if obj_color == 'w':
        k = 0.2
    else:
        k = -0.2
    m = 1

    height, width = grayscale_img.shape
    output_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            y_start, y_end = max(0, y - m), min(height, y + m + 1)
            x_start, x_end = max(0, x - m), min(width, x + m + 1)
            grid = grayscale_img[y_start:y_end, x_start:x_end]
            mu = np.mean(grid)
            s = np.std(grid)
            threshold = mu + k*s

            if grayscale_img[y, x] < threshold:
                output_image[y, x] = 0
            else:
                output_image[y, x] = 225

    return output_image


def sauvola(grayscale_img):
    k, m = 0.2, 1

    height, width = grayscale_img.shape
    output_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            y_start, y_end = max(0, y - m), min(height, y + m + 1)
            x_start, x_end = max(0, x - m), min(width, x + m + 1)
            grid = grayscale_img[y_start:y_end, x_start:x_end]
            mu = np.mean(grid)
            s = np.std(grid)
            threshold = mu*(1 - k*(1 - s/128))

            if grayscale_img[y, x] < threshold:
                output_image[y, x] = 0
            else:
                output_image[y, x] = 225

    return output_image


def christian(grayscale_img):
    m, k = 1, 0.5
    height, width = grayscale_img.shape
    output_image = np.zeros((height, width), dtype=np.uint8)
    min_br = np.min(grayscale_img)
    std_list = []

    for y in range(height):
        for x in range(width):
            y_start, y_end = max(0, y - m), min(height, y + m + 1)
            x_start, x_end = max(0, x - m), min(width, x + m + 1)
            grid = grayscale_img[y_start:y_end, x_start:x_end]
            std_list.append(np.std(grid))

    max_std = max(std_list)

    for y in range(height):
        for x in range(width):
            y_start, y_end = max(0, y - m), min(height, y + m + 1)
            x_start, x_end = max(0, x - m), min(width, x + m + 1)
            grid = grayscale_img[y_start:y_end, x_start:x_end]
            mu = np.mean(grid)
            s = np.std(grid)
            threshold = (1-k) * mu+k*min_br+k*(s/max_std) * (mu-min_br)

            if grayscale_img[y, x] < threshold:
                output_image[y, x] = 0
            else:
                output_image[y, x] = 225

    return output_image
