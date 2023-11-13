import torch
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
import argparse
import cv2
import numpy as np
from sam_lora import LoRA_Sam

# 设置一些参数，包括模型的路径
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace()
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/best_nofocal.pth"
# 开始载入模型
model = sam_model_registry["vit_b"](args)
lora_sam = LoRA_Sam(model,16).to(device)
with open(args.sam_checkpoint, "rb") as f:
    state_dict = torch.load(f,map_location=device)
    lora_sam.sam.load_state_dict(state_dict['model'])
# 载入完成
# 先设置模型为eval模式，并得到预测器
lora_sam.sam.eval()
predictor = SammedPredictor(lora_sam.sam)

# 正常情况下，这之前的代码都不会报错，你只需要处理后面的代码就可以了


# 开始载入图像和掩码，并对掩码作预处理
image = cv2.imread('data_demo/images/amos_0004_75.png')
mask_truth = cv2.imread('data_demo/masks/amos_0004_75_aorta_000.png', cv2.IMREAD_GRAYSCALE)
mask_truth = mask_truth/255
# 开始预测
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    multimask_output=True,
)
# 输出结果
cv2.imwrite('amos_0507_31_mask.jpg', masks[0]*255)