import cv2
import numpy as np

def pad_image(image_path, patch_height=256, patch_width=256):
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 计算需要填充的像素数量
    pad_height = patch_height - (height % patch_height)
    pad_width = patch_width - (width % patch_width)

    # 进行填充操作
    padded_image = cv2.copyMakeBorder(image, pad_height // 2, pad_height - (pad_height // 2), 
                                      pad_width // 2, pad_width - (pad_width // 2), 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

if __name__ == '__main__':
    image_path = 'data/eyes/image/82001020010_20200102090140762.jpg'
    padded_image = pad_image(image_path)
    print(padded_image.shape)
    cv2.imwrite('test_padding.jpg', padded_image)