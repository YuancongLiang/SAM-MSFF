import torch
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from sam_lora import LoRA_Sam
import argparse
from Dataset import DriveDataset
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image

# 设置一些参数，包括模型的路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace()
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model\\best_nofocal.pth"
args.data_dir = "data\\drive"
args.batch_size = 32
args.num_epochs = 10
args.learning_rate = 0.001


class ImageDataset(Dataset):
    def __init__(self, root_dir, mode='training', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.input_images = os.listdir(os.path.join(self.root_dir, 'images'))
        print(self.input_images)
        self.target_images = os.listdir(os.path.join(self.root_dir, '1st_manual'))
        print(self.target_images)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_path = os.path.join(self.root_dir, 'images', self.input_images[idx])
        target_image_path = os.path.join(self.root_dir, '1st_manual', self.target_images[idx])

        input_image = cv2.imread(input_image_path)

        if input_image is None:
            print(input_image_path)
            raise FileNotFoundError(f"未能在'{input_image_path}'加载input_image")

        target_image = Image.open(target_image_path)
        target_image =  cv2.cvtColor(np.asarray(target_image),cv2.COLOR_RGB2BGR)

        if target_image is None:
            raise FileNotFoundError(f"未能在'{target_image_path}'加载target_image")


        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


transformations = transforms.Compose([
    transforms.ToTensor(),  # 转换为 Tensor
])


# 定义训练数据集和测试数据集
train_dataset = ImageDataset(args.data_dir,mode='training',transform=transformations)
#test_dataset = DriveDataset(args.data_dir, image_size=args.image_size, mode='test', requires_name=False)


# 计算训练集的像素均值和标准差
#train_dataset.get_mean_std()

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 开始载入模型
model = sam_model_registry["vit_b"](args)
lora_sam = LoRA_Sam(model, 16).to(device)
with open(args.sam_checkpoint, "rb") as f:
    state_dict = torch.load(f, map_location=device)
    lora_sam.sam.load_state_dict(state_dict['model'])
# 载入完成
# 设置模型为训练模式
lora_sam.sam.train()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lora_sam.parameters(), lr=args.learning_rate)

# 训练模型
for epoch in range(args.num_epochs):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
        images, masks = batch
        images = images.to(device)  # 将输入图像转移到设备上
        masks = masks.to(device)  # 将目标图像转移到设备上

        optimizer.zero_grad()
        predictor = SammedPredictor(lora_sam.sam)
        predictor.set_image(images.cpu())
        masks_pred, scores, logits = predictor.predict(
            multimask_output=True,
        )

        # 计算损失
        loss = criterion(masks_pred, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 输出每个epoch的训练损失
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{args.num_epochs} Train Loss: {avg_train_loss:.4f}")

