import cv2
import numpy as np
from Dataset import EyesDataset
from tqdm import tqdm

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

def adjust_image(image_path, patch_height, patch_width):
    image = pad_image(image_path, patch_height, patch_width)
    height, width, _ = image.shape


    # 删除像素
    adjusted_image = image[patch_height//2:-patch_height//2, patch_width//2:-patch_width//2]
    #cv2.imwrite('data/82001020010_20200102090140762_adjusted.jpg', adjusted_image)
    return adjusted_image


def split_image(image_path:str, save_path:str, patch_height=256, patch_width=256):
    # 读取图像
    image = adjust_image(image_path, patch_height, patch_width)
    image_name = image_path.split('/')[-1].split('.')[0]
    height, width, _ = image.shape

    # 定义小图尺寸和步长
    stride = patch_height//2
    patches = []
    for y in range(0, height-stride, stride):
        for x in range(0, width-stride, stride):
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)

    # 保存小图
    for i, patch in enumerate(patches):
        # 保存小图
        if i!= 0 and i!=12 and i!= 156 and i!=168:
            cv2.imwrite(f"{save_path}/{image_name}_patch_{i}.png", patch)

if __name__ == '__main__':
    train_dataset = EyesDataset("data/eyes_selected", image_size=256, mode='train', point_num=1, mask_num=1, requires_name = False)
    #adjust_image(image_path, patch_height=256, patch_width=256)
    for image_path in tqdm(train_dataset.label_paths):
        split_image(image_path=image_path, save_path='data/eyes_patch/mask', patch_height=256, patch_width=256)