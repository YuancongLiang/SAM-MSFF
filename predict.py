import torch
from torch.nn import functional as F
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed_hq import SammedPredictorHQ
from segment_anything.predictor_sammed_fusion import SammedPredictorFusion
from segment_anything.predictor_sammed import SammedPredictor
import argparse
import cv2
import numpy as np
from Dataset import StareDataset, FivesDataset, Chasedb1Dataset, DriveDataset, EyesDataset, RBVDataset, HRFDataset, Drive2Dataset, HAGISDataset, LESDataset
from sam_lora import LoRA_Sam
from math import ceil
from tqdm import tqdm
from metric import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="if use adapter in encoder")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/rank16.pth", help="the checkpoint path of sam model")
    parser.add_argument("--patch_size", type=int, default=64, help="patch size in betti number calculation")
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    return args
np.random.seed(20240328)
class ImageProcessing:
    def __init__(self, model, image=None, patch_height=256, patch_width=256, stride=128):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride
        self.original_image = image
        self.transformed_image = image
        self.transform_info = None
        self.split_image_list = None
        self.model = model

    def pad_image(self):
        pad_height = (self.patch_height - (self.original_height % self.patch_height)) % self.stride
        pad_width = (self.patch_width - (self.original_width % self.patch_width)) % self.stride
        
        self.transform_info = (pad_height // 2, pad_height - (pad_height // 2),pad_width // 2, pad_width - (pad_width // 2))
        pad_height = pad_height + (self.patch_height-self.stride)
        pad_width = pad_width + (self.patch_width-self.stride)
        self.transformed_image = cv2.copyMakeBorder(self.original_image, pad_height // 2, pad_height - (pad_height // 2),
                                                pad_width // 2, pad_width - (pad_width // 2), 
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    def get_transformed_image(self):    # test only
        return self.transformed_image
    
    def set_transformed_image(self, image):    
        self.transformed_image = image

    def set_original_image(self,image):
        self.original_image = image
        self.original_height, self.original_width, _ = self.original_image.shape

    def set_patch_mask_list(self, patch_mask_list):
        self.patch_mask_list = patch_mask_list

    def restore_image(self):
        if self.transform_info is not None:
            padded_image_height , padded_image_width = self.transformed_image.shape
            pad_height_up, pad_height_down,pad_width_left,pad_width_right = self.transform_info
            self.transformed_image = self.transformed_image[pad_height_up:padded_image_height-pad_height_down, pad_width_left:padded_image_width-pad_width_right]
            self.transform_info = None
        else:
            pass
            # print("You haven't made any changes yet")
    
    def split_image(self):
        if self.transform_info is not None:
            image = self.transformed_image
            height, width, _ = image.shape
            patches = []
            for y in range(0, height-self.stride, self.stride):
                for x in range(0, width-self.stride, self.stride):
                    patch = image[y:y+self.patch_height, x:x+self.patch_width]
                    patches.append(patch)
        else:
            print("You can't split the image before padding it")

        return patches
    
    def extract_center_region(self, patch):
        # 计算中心区域起始位置
        start_y = (patch.shape[0] - self.stride) // 2
        start_x = (patch.shape[1] - self.stride) // 2

        # 提取中心区域
        center_region = patch[start_y:start_y+self.stride, start_x:start_x+self.stride]

        return center_region
    
    def merge_patches(self):
        # 计算大图尺寸
        patches = [self.extract_center_region(split_image) for split_image in self.patch_mask_list]
        patch_height = patches[0].shape[0]
        patch_width = patches[0].shape[1]
        num_rows = ceil(self.original_height / self.stride)
        num_cols = ceil(self.original_width / self.stride)
        image_height = patch_height * num_rows
        image_width = patch_width * num_cols

        # 创建大图空白画布
        merged_image = np.zeros((image_height, image_width), dtype=np.uint8)

        # 遍历小图
        index = 0
        for row in range(num_rows):
            for col in range(num_cols):
                y = row * patch_height
                x = col * patch_width
                merged_image[y:y+patch_height, x:x+patch_width] = patches[index]
                index += 1
        self.set_transformed_image(merged_image)
        self.restore_image()
        return self.transformed_image
    
    def predict(self):
        if self.original_image is not None:
            self.pad_image()
            split_image_list = self.split_image()
            mask_list = []
            logits_list = []
            for image in split_image_list:
                self.model.set_image(image)
                logits, scores, _ = self.model.predict(
                    multimask_output=True,
                    # hq_token_only=False,
                    return_logits=True,
                )
                logits_list.append(logits[0]) 
                sigmoid_output = torch.sigmoid(torch.tensor(logits))
                masks = (sigmoid_output > 0.5).float().numpy()
                mask_list.append(masks[0])
            self.set_patch_mask_list(mask_list)
            predicted_mask = self.merge_patches()
            self.set_patch_mask_list(logits_list)
            predicted_logit = self.merge_patches()
        
        else:
            print('you should set the original image first')
        return predicted_mask, predicted_logit

if __name__ == '__main__':
    # 设置一些参数，包括模型的路径
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 开始载入模型
    # model = sam_model_registry["vit_b_hq"](args)
    model = sam_model_registry["vit_b"](args)
    # model = sam_model_registry["vit_b_fusion"](args)
    lora_sam = LoRA_Sam(model,16).to(device)
    with open(args.sam_checkpoint, "rb") as f:
        state_dict = torch.load(f,map_location=device)
        lora_sam.sam.load_state_dict(state_dict['model'], strict=False)
    # 载入完成
    # 先设置模型为eval模式，并得到预测器
    lora_sam.sam.eval()
    # predictor = SammedPredictorHQ(lora_sam.sam)
    predictor = SammedPredictor(lora_sam.sam)
    # predictor = SammedPredictorFusion(lora_sam.sam)
    image_processor = ImageProcessing(predictor)


    # dataset = FivesDataset("data/FIVES", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    # dataset = Chasedb1Dataset("data/chasedb1", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    # dataset = StareDataset("data/stare", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    # dataset = HRFDataset("data/HRF", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    # dataset = RBVDataset("data/RetinaBloodVessel", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    # dataset = Drive2Dataset("data/DRIVE2", image_size=256, mode='test', point_num=1, mask_num=1, requires_name = False)
    dataset = DriveDataset("data/DRIVE", image_size=256, mode='train', point_num=1, mask_num=1, requires_name = False)
    iou_list = []
    dice_list = []
    cldice_list = []
    auc_list = []
    acc_list = []
    hd_list = []
    betti0_list = []
    betti1_list = []
    for index, image_path in enumerate(dataset.image_paths):
        image = cv2.imread(image_path)
        image_processor.set_original_image(image)
        predicted_mask, predicted_iou = image_processor.predict()
        mask_truth = cv2.imread(dataset.label_paths[index], cv2.IMREAD_GRAYSCALE)
        mask_truth = mask_truth/255
        iou_list.append(calculate_iou(mask_truth, predicted_mask))
        dice_list.append(calculate_dice_score(mask_truth, predicted_mask))
        cldice_list.append(calculate_cldice(mask_truth, predicted_mask))
        auc_list.append(calculate_auc(mask_truth, predicted_mask))
        acc_list.append(calculate_acc(mask_truth, predicted_mask))
        hd_list.append(calculate_hausdorff(mask_truth, predicted_mask))
        betti0 , betti1 = calculate_betti_numbers(mask_truth, predicted_mask, patch_size=args.patch_size)
        betti0_list.append(betti0)
        betti1_list.append(betti1)
        image_name = image_path.split('/')[-1]
        print(f'the file name is:{image_name}')
        print(f'the iou is:{str(iou_list[-1])}')
        print(f'the dice score is:{str(dice_list[-1])}')
        print(f'the cldice is:{str(cldice_list[-1])}')
        print(f'the auc is:{str(auc_list[-1])}')
        print(f'the acc is:{str(acc_list[-1])}')
        print(f'the hd is:{str(hd_list[-1])}')
        print(f'the betti0 is:{str(betti0_list[-1])}')
        print(f'the betti1 is:{str(betti1_list[-1])}')
    print(f'the mean iou is:{str(np.mean(iou_list))}±{str(np.std(iou_list))}')
    print(f'the mean dice score is:{str(np.mean(dice_list))}±{str(np.std(dice_list))}')
    print(f'the mean cldice is:{str(np.mean(cldice_list))}±{str(np.std(cldice_list))}')
    print(f'the mean auc is:{str(np.mean(auc_list))}±{str(np.std(auc_list))}')
    print(f'the mean acc is:{str(np.mean(acc_list))}±{str(np.std(acc_list))}')
    print(f'the mean hd is:{str(np.mean(hd_list))}±{str(np.std(hd_list))}')
    print(f'the mean betti0 is:{str(np.mean(betti0_list))}±{str(np.std(betti0_list))}')
    print(f'the mean betti1 is:{str(np.mean(betti1_list))}±{str(np.std(betti1_list))}')
