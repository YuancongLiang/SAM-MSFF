import os
import shutil

def copy_files(source_folder, destination_folder):
    # 遍历源文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 构建源文件的完整路径
            source_file = os.path.join(root, file)

            # 构建目标文件的完整路径
            destination_file = os.path.join(destination_folder, file)

            # 复制文件
            shutil.copy2(source_file, destination_file)

# 示例用法
source_folder = "/home/liangyuancong/Pytorch-resnet/data/person/"  # 替换为源文件夹的路径
destination_folder = "/home/liangyuancong/SAM-Med2D/data/eyes_all/image/"  # 替换为目标文件夹的路径

copy_files(source_folder, destination_folder)