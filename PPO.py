import torch
import torch.nn.functional as F
import numpy as np
import argparse
from sam_lora import LoRA_Sam
from segment_anything import sam_model_registry
import cv2
from Dataset import EyesDataset, stack_dict_batched
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
torch.manual_seed(3407)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="lora_patch", help="run model name")
    parser.add_argument("--epochs", type=int, default=90, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/eyes_patch", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--lr", type=float, default=1e-8, help="learning rate")
    parser.add_argument("--resume", type=str, default='/home/liangyuancong/SAM-Med2D/pretrain_model/patch.pth', help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/pretrained_lora.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=True, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--workers", type=int, default=0, help="amount of workers")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

# 将数据批量导入device，batch_input是一个字典，保存了image和label的键值对
def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label': # 如果是image和label就导入到device
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size: # 如果是
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

class ActionValueModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        image_embeddings = self.model.image_encoder(x)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
        if args.multimask:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res = []
            for i, idx in enumerate(max_indexs):
                low_res.append(low_res_masks[i:i+1, idx])
            low_res_masks = torch.stack(low_res, 0)
        masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks)
        return masks, iou_predictions

class RewardModel(nn.Module):
    pass

class GAE(nn.Module):
    pass

if __name__ == '__main__':
    args = parse_args()
    sam = sam_model_registry[args.model_type](args)
    lora_sam = LoRA_Sam(sam,r = 16).to(args.device)
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            lora_sam.sam.load_state_dict(checkpoint['model'])
            print(f"*******load {args.resume}")
    # 载入完成
    lora_sam.sam.eval()
    train_dataset = EyesDataset("data/eyes", image_size=256, mode='test', requires_name=False, point_num=1, mask_num=1)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    action_model = ActionValueModel(lora_sam.sam)
    for batched_input in tqdm(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        masks, value = action_model(batched_input["image"])
        print(masks)
