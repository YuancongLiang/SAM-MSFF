import os
from tqdm import tqdm 
import time

def delete_small_png_files(folder_path):
    deleted_files = []  # 用于保存已删除的文件名
    # 遍历文件夹下的所有文件
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.png'):  # 仅处理以 .png 结尾的文件
            file_path = os.path.join(folder_path, file_name)
            file_size = os.path.getsize(file_path)  # 获取文件大小（字节数）

            if file_size < 10 * 1024:  # 如果文件大小小于 10KB
                os.remove(file_path)  # 删除文件
                image_file_path = file_path.replace('Original','Ground truth')
                time.sleep(0.001)
                os.remove(image_file_path)
                deleted_files.append(image_file_path)  # 将删除的文件名添加到列表中
    return deleted_files

# 示例用法
folder_path = "/home/liangyuancong/SAM-Med2D/data/fives_patch/train/Original/"  # 替换为你的文件夹路径
deleted_files = delete_small_png_files(folder_path)