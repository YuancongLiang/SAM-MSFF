from segment_anything import build_sam, SamAutomaticMaskGenerator 
from segment_anything import sam_model_registry
import time
import torch.nn as nn
from torchvision.transforms import v2
from torch.nn.parallel import DataParallel
from sam_lora import LoRA_Sam
import torch
import numpy as np
import os
import datetime
from loss.cldice import soft_cldice, soft_dice_cldice, dice_ce
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from Dataset import TrainingDataset, stack_dict_batched, StareDataset, EyesDataset, FivesDataset, Chasedb1Dataset
from torch import optim
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
import argparse
from tqdm import tqdm
from torchviz import make_dot
from metrics import SegMetrics
import random
# from torchsummary import summary
from torch.nn import functional as F
# torch.manual_seed(3407)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="attention_pointconv", help="run model name")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/fives_patch", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device_ids', type=list, default=[0,1,2,3])
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default='pretrain_model/attention_conv.pth', help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b_fusion", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/pretrained_lora.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=4, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=False, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--workers", type=int, default=0, help="amount of workers")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

class LoRAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hq_token_only= False
    def forward(self, x):
        image_embeddings = self.model.image_encoder(x)
        interm_embeddings = self.model.image_encoder.get_interm_embeddings()
        interm_embeddings = interm_embeddings[0] # early layer
        patch_embeddings = self.model.image_encoder.get_patch_embedding()
        # patch_embeddings = self.model.image_encoder.get_image()
        outputs = torch.tensor([]).to(x.device)
        for curr_embedding, curr_interm, patch_embedding in zip(image_embeddings, interm_embeddings,patch_embeddings):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            # dense_embeddings = torch.zeros_like(dense_embeddings).to(dense_embeddings.device)
            low_res_masks, iou_predictions = self.model.mask_decoder(
                # image_embeddings = image_embeddings,
                image_embeddings = curr_embedding.unsqueeze(0),
                image_pe = self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=args.multimask,
                hq_token_only=self.hq_token_only,
                interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0),
                patch_embeddings=patch_embedding.unsqueeze(0),
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
            outputs = torch.cat([outputs,masks])
        return outputs, iou_predictions

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

@torch.no_grad()
def val_one_epoch(args, model, optimizer, val_loader, epoch, criterion):
    print("start to validate")
    val_loader = tqdm(val_loader)
    val_losses = []
    model.eval()
    for batch, batched_input in enumerate(val_loader):
        
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        labels = batched_input["label"]
        masks, iou_predictions = model(batched_input["image"])
        loss = criterion(masks, labels, iou_predictions)
        gpu_info = {}
        gpu_info['gpu_name'] = model.device_ids
        val_loader.set_postfix(val_loss=loss.item(), gpu_info=gpu_info)
        val_losses.append(loss.item())
    return val_losses


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    train_losses = []
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        labels = batched_input["label"]
        images = batched_input["image"]
        masks, iou_predictions = model(images)
        loss = criterion(masks, labels, iou_predictions)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        gpu_info = {}
        gpu_info['gpu_name'] = model.device_ids
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)
        train_losses.append(loss.item())
    return train_losses



if __name__ == '__main__':
    args = parse_args()
    seed = torch.initial_seed()
    print("Random Seed:", seed)
    sam = sam_model_registry[args.model_type](args)
    criterion = soft_dice_cldice(iter_=80, alpha=0.5)
    # criterion = GC_2D(lmda=5)
    lora_sam = LoRA_Sam(sam,r = 16).to(args.device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, lora_sam.sam.parameters()), lr=args.lr)
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            sam.load_state_dict(checkpoint['model'],strict=False)
            print(f"*******load {args.resume}")
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')
    for n, value in lora_sam.sam.image_encoder.named_parameters():
        if "linear_" in n:
            value.requires_grad = True
        elif "qkv.qkv.bias" in n:
            value.requires_grad = True
        elif "proj.proj.bias" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False
    for n, value in lora_sam.sam.named_parameters():
        if "image_encoder" in n:
            pass
        else:
            value.requires_grad = True
    # summary(lora_sam.sam.image_encoder, (3, 256, 256), device='cuda')
    # for n, value in lora_sam.sam.named_parameters():
    #     print(n, value.requires_grad)
    #train_dataset1 = StareDataset("data/stare_patch", image_size=256, mode='train', requires_name=False, point_num=1, mask_num=args.mask_num)
    #train_dataset2 = Chasedb1Dataset("data/chasedb1_patch", image_size=256, mode='train', requires_name=False, point_num=1, mask_num=args.mask_num)
    train_dataset = FivesDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    # train_dataset2 = FivesDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False, require_torch_augment=True)
    #train_dataset = EyesDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = StareDataset('data/stare', image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = TrainingDataset('data_demo', image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    # train_dataset = ConcatDataset([train_dataset1,train_dataset2])
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataset = FivesDataset(args.data_path, image_size=args.image_size, mode='test', point_num=1, mask_num=args.mask_num, requires_name = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    print('*******Train data:', len(train_dataset))
    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    model = DataParallel(LoRAModel(lora_sam.sam),device_ids=args.device_ids)
    best_loss = 1e10
    l = len(train_loader)
    
    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)
        val_losses = val_one_epoch(args, model, optimizer, test_loader, epoch, criterion)
        if args.lr_scheduler:
            scheduler.step()
        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler else args.lr
        val_average_losses = np.mean(val_losses)
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f},Val loss:{val_average_losses:.4f}")
        save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
        state = {'model': lora_sam.sam.float().state_dict(), 'optimizer': optimizer}
        torch.save(state, save_path)

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))