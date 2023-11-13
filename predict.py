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

def calculate_iou(true_mask, predicted_mask):
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_score(true_mask, predicted_mask):
    intersection = np.logical_and(true_mask, predicted_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(true_mask) + np.sum(predicted_mask))
    return dice_score

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

image = cv2.imread('data/stare/image/im0001.ppm')
mask_truth = cv2.imread('data/stare/mask/im0001.vk.ppm', cv2.IMREAD_GRAYSCALE)
mask_truth = mask_truth/255
print(image.shape)
print(mask_truth.shape)
predictor.set_image(image)
input_point = np.array([[162, 127]])
input_label = np.array([0])
masks, scores, logits = predictor.predict(
    # point_coords=input_point,
    # point_labels=input_label,
    multimask_output=True,
)
print('the predicted scores is '+str(scores))
print('the iou is ' + str(calculate_iou(mask_truth, masks[0])))
print('the dice is ' + str(calculate_dice_score(mask_truth, masks[0])))
cv2.imwrite('test_mask.jpg', masks[0]*255)
dataset = StareDataset("data/stare", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1)
print('start to predict dataset')
iou_list = []
dice_list = []
for index, image_path in enumerate(dataset.image_paths):
    image = cv2.imread(image_path)
    mask_truth = cv2.imread(dataset.label_paths[index], cv2.IMREAD_GRAYSCALE)
    mask_truth = mask_truth/255
    predictor.set_image(image)
    masks_pred, scores, logits = predictor.predict(
        multimask_output=True,
    )
    iou_list.append(calculate_iou(mask_truth, masks_pred[0]))
    dice_list.append(calculate_dice_score(mask_truth, masks_pred[0]))
    print('the file name is:'+image_path.split('/')[-1])
    print('the iou is:'+str(iou_list[-1]))
    print('the dice score is:'+str(dice_list[-1]))
print('the mean iou is:'+str(np.mean(iou_list)))
print('the mean dice score is:'+str(np.mean(dice_list)))