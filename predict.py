import torch
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
import argparse
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="if use adapter in encoder")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="the checkpoint path of sam model")

def calculate_iou(true_mask, predicted_mask):
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_score(true_mask, predicted_mask):
    intersection = np.logical_and(true_mask, predicted_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(true_mask) + np.sum(predicted_mask))
    return dice_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
image = cv2.imread('test.jpg')
# image = cv2.imread('data_demo/images/amos_0507_31.png')
print(image.shape)
args = argparse.Namespace()
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"
model = sam_model_registry["vit_b"](args).to(device)
predictor = SammedPredictor(model)
predictor.set_image(image)
input_point = np.array([[162, 127]])
input_label = np.array([0])
masks, scores, logits = predictor.predict(
    # point_coords=input_point,
    # point_labels=input_label,
    multimask_output=True,
)
print(masks.shape)
cv2.imwrite('test_mask.jpg', masks[0]*255)