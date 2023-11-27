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
import cv2

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

# 定义训练数据集和测试数据集
train_dataset = DriveDataset(args.data_dir, image_size=args.image_size, mode='training', requires_name=False)
#test_dataset = DriveDataset(args.data_dir, image_size=args.image_size, mode='test', requires_name=False)

print(train_dataset)

# 计算训练集的像素均值和标准差
train_dataset.get_mean_std()

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
        images = batch["image"].to(device)
        masks = batch["label"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)

        optimizer.zero_grad()
        predictor = SammedPredictor(lora_sam.sam)
        predictor.set_image(images[0][0].cpu())
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

