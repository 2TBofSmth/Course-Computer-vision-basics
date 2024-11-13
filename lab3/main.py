import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *


img_paths = [
    ['data/input_img1.png', 'data/input_img2.png', 'data/input_img3.png'],
    ['data/gr_img1.png', 'data/gr_img2.png', 'data/gr_img3.png'],
    ['data/otsu_img1.png', 'data/otsu_img2.png', 'data/otsu_img3.png'],
    ['data/niblek_img1.png', 'data/niblek_img2.png', 'data/niblek_img3.png'],
    ['data/sauvola_img1.png', 'data/sauvola_img2.png', 'data/sauvola_img3.png'],
    ['data/christain_img1.png', 'data/christain_img2.png', 'data/christain_img3.png']
]

row_titles = [
    'Input image',
    'Grayscaled image',
    'Otsu method',
    'Niblek method',
    'Sauvola method',
    'Christian method'
]


def create_imgs(img_path, n):
    img = cv2.imread(img_path)
    gr_img = to_grayscale(img)
    cv2.imwrite(f'data/gr_img{n}.png', gr_img)
    cv2.imwrite(f'data/otsu_img{n}.png', otsu(gr_img))
    cv2.imwrite(f'data/niblek_img{n}.png', niblek(gr_img, 'b'))
    cv2.imwrite(f'data/sauvola_img{n}.png', sauvola(gr_img))
    cv2.imwrite(f'data/christain_img{n}.png', christian(gr_img))


for i in range(1, 4):
    create_imgs(img_paths[0][i-1], i)

fig, axes = plt.subplots(6, 3, figsize=(10, 15))
fig.suptitle('Comparison of different binarization methods', fontsize=16)
plt.subplots_adjust(left=0.2, wspace=0.1, hspace=0.5)

for i in range(6):
    for j in range(3):
        image = mpimg.imread(img_paths[i][j])
        axes[i, j].imshow(image, cmap='gray')
        axes[i, j].axis('off')
        axes[i, j].set_title(f'{row_titles[i]}', fontsize=8)

plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
plt.show()
