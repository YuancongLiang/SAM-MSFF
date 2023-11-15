import torch
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
import argparse
import cv2
import numpy as np
from Dataset import StareDataset
from sam_lora import LoRA_Sam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="if use adapter in encoder")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="the checkpoint path of sam model")

def calculate_iou(true_mask, predicted_mask) -> np.ndarray:
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_score(true_mask, predicted_mask) -> np.ndarray:
    intersection = np.logical_and(true_mask, predicted_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(true_mask) + np.sum(predicted_mask))
    return dice_score

# 设置一些参数，包括模型的路径
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace()
args.image_size = 256
args.encoder_adapter = True
# args.sam_checkpoint = "pretrain_model/best_nofocal.pth"
args.sam_checkpoint = "workdir/models/lora_patch/epoch2_sam.pth"
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