import torch
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
import argparse
import cv2
import numpy as np
from Dataset import StareDataset, FivesDataset, Chasedb1Dataset, DriveDataset, EyesDataset
from sam_lora import LoRA_Sam
from math import ceil
from tqdm import tqdm

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

def calculate_acc(true_mask, predicted_mask) -> np.ndarray:
    # 将掩码转换为布尔型数组
    gt_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(predicted_mask).astype(bool)
    # 计算掩码相等的像素数量
    num_correct_pixels = np.sum(gt_mask == pred_mask)
    # 计算总像素数量
    total_pixels = np.prod(gt_mask.shape)
    # 计算准确率
    accuracy = num_correct_pixels / total_pixels
    return accuracy


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
            print("You haven't made any changes yet")
    
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
            scores_list = []
            for image in split_image_list:
                self.model.set_image(image)
                masks, scores, logits = self.model.predict(
                    multimask_output=True,
                )
                mask_list.append(masks[0])   
                scores_list.append(scores) 
            self.set_patch_mask_list(mask_list)
            predicted_mask = self.merge_patches()
            iou_scores = np.array(scores_list).mean()
        
        else:
            print('you should set the original image first')
        return predicted_mask, iou_scores


# 设置一些参数，包括模型的路径
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace()
args.image_size = 256
args.encoder_adapter = True
# args.sam_checkpoint = "workdir/models/fives_patch/epoch2_sam.pth"
args.sam_checkpoint = "workdir/models/stare_chasedb1_patch/epoch20_sam.pth"
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
image_processor = ImageProcessing(predictor)

image = cv2.imread('/home/liangyuancong/SAM-Med2D/3_A_origin.png')
mask_truth = cv2.imread('/home/liangyuancong/SAM-Med2D/3_A_mask.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('/home/liangyuancong/SAM-Med2D/im0001_image.jpg')
# mask_truth = cv2.imread('/home/liangyuancong/SAM-Med2D/data/stare/mask/im0001.vk.ppm', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('/home/liangyuancong/SAM-Med2D/data/chasedb1/Image_14L.jpg')
# mask_truth = cv2.imread('/home/liangyuancong/SAM-Med2D/data/chasedb1/Image_14L_1stHO.png', cv2.IMREAD_GRAYSCALE)

mask_truth = mask_truth/255
image_processor.set_original_image(image)
predicted_mask,predicted_iou = image_processor.predict()
print('the predicted iou is ' + str(predicted_iou))
print('the iou is ' + str(calculate_iou(mask_truth, predicted_mask)))
print('the dice is ' + str(calculate_dice_score(mask_truth, predicted_mask)))
print('the acc is ' + str(calculate_acc(mask_truth, predicted_mask)))
cv2.imwrite('3_A_predicted.png', predicted_mask*255)

# dataset = StareDataset("data/stare", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1)
# dataset = Chasedb1Dataset("data/chasedb1", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1)
# dataset = FivesDataset("data/FIVES", image_size=256, mode='test', requires_name=True, point_num=1, mask_num=1)
# dataset = DriveDataset("data/drive", image_size=256, mode='test', requires_name=True, point_num=1, mask_num=1)
dataset = EyesDataset("data/eyes_all", image_size=256, mode='test', requires_name=True, point_num=1, mask_num=1)
print('start to predict dataset')
for index, image_path in enumerate(tqdm(dataset.image_paths)):
    image = cv2.imread(image_path)
    image_processor.set_original_image(image)
    predicted_mask, predicted_iou= image_processor.predict()
    cv2.imwrite(image_path.replace('image/','mask/Result_').replace('jpg','png'),predicted_mask*255)
# iou_list = []
# dice_list = []
# acc_list = []
# for index, image_path in enumerate(dataset.image_paths):
#     image = cv2.imread(image_path)
#     image_processor.set_original_image(image)
#     predicted_mask, predicted_iou = image_processor.predict()
#     mask_truth = cv2.imread(dataset.label_paths[index], cv2.IMREAD_GRAYSCALE)
#     mask_truth = mask_truth/255
#     iou_list.append(calculate_iou(mask_truth, predicted_mask))
#     dice_list.append(calculate_dice_score(mask_truth, predicted_mask))
#     acc_list.append(calculate_acc(mask_truth, predicted_mask))
#     print('the file name is:'+image_path.split('/')[-1])
#     print('the iou is:'+str(iou_list[-1]))
#     print('the dice score is:'+str(dice_list[-1]))
#     print('the acc is:'+str(acc_list[-1]))
# print('the mean iou is:'+str(np.mean(iou_list)))
# print('the mean dice score is:'+str(np.mean(dice_list)))
# print('the mean acc is:'+str(np.mean(acc_list)))
