from segment_anything import build_sam, SamAutomaticMaskGenerator 
from segment_anything import sam_model_registry
import time
from torch.nn.parallel import DataParallel
# from bitsandbytes import BitsAndBytesConfig
# from transformers import AutoModelForCausalLM
from sam_lora import LoRA_Sam
import torch
import numpy as np
import os
import datetime
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from Dataset import TrainingDataset, stack_dict_batched, StareDataset, EyesDataset, FivesDataset, Chasedb1Dataset
from torch import optim
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
import argparse
from tqdm import tqdm
from metrics import SegMetrics
import random
from torch.nn import functional as F
# torch.manual_seed(3407)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="8_patch", help="run model name")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/fives_patch", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default='/home/lyc/SAMMed-LoRA/pretrain_model/sam-med2d_b.pth', help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/pretrained_lora.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=4, help="point iterations")
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

def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    # 如果有point坐标提示
    if  batched_input["point_coords"] is not None:
        # 记录下坐标和label
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    # 是否无梯度
    if decoder_iter:
        with torch.no_grad():
            # 扔进model中，返回稀疏特征和密集特征
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
    else:
        # 扔进model中，返回稀疏特征和密集特征
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )
    
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
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

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

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
        image_embeddings = model.image_encoder(batched_input["image"])
        batch, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(batch):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
        loss = criterion(masks, labels, iou_predictions)

    return loss


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    model.train()
    train_loader = tqdm(train_loader)
    train_losses = []
    
    for batch, batched_input in enumerate(train_loader):
        
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        # 冻结除了lora层以外的参数
        # for n, value in model.sam.image_encoder.named_parameters():
        for n, value in model.image_encoder.named_parameters():
            if "linear_" in n:
                value.requires_grad = True
            elif "qkv.qkv.bias" in n:
                value.requires_grad = True
            elif "proj.proj.bias" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        # for n, value in lora_sam.sam.image_encoder.named_parameters():
        #     if "Adapter" in n:
        #         value.requires_grad = True
        #     else:
        #         value.requires_grad = False
        labels = batched_input["label"]
        image_embeddings = model.image_encoder(batched_input["image"])
        batch, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(batch):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
        loss = criterion(masks, labels, iou_predictions)
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first mask prompt: {SegMetrics(masks, labels, args.metrics)}')

        point_num = random.choice(args.point_list)
        # batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = setting_prompt_none(batched_input)
        # 如果我们对血管进行分割，是不应该有点提示的，这里应该将点提示删掉
        batched_input = to_device(batched_input, args.device)
        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True
        # 后面不对encoder进行训练
        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            train_iter_metrics = [0] * len(args.metrics)
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            # if iter != args.iter_point - 1:
            #     point_num = random.choice(args.point_list)
            #     batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
            #     batched_input = to_device(batched_input, args.device)
           
            if int(batch+1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')
            if int(batch+1) % 200 == 0:
                print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
                save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
                state = {'model': model.state_dict(), 'optimizer': optimizer}
                torch.save(state, save_path)

            train_losses.append(loss.item())

            gpu_info = {}
            gpu_info['gpu_name'] = args.device 
            train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

            train_batch_metrics = SegMetrics(masks, labels, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
    return train_losses, train_iter_metrics

if __name__ == '__main__':
    args = parse_args()
    sam = sam_model_registry[args.model_type](args)
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=args.lr)
    criterion = FocalDiceloss_IoULoss(weight=0.0,iou_scale=0)
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            sam.load_state_dict(checkpoint['model'])
            # sam.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")
    lora_sam = LoRA_Sam(sam,r = 8).to(args.device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, lora_sam.sam.parameters()), lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')
    #train_dataset1 = StareDataset("data/stare_patch", image_size=256, mode='train', requires_name=False, point_num=1, mask_num=args.mask_num)
    #train_dataset2 = Chasedb1Dataset("data/chasedb1_patch", image_size=256, mode='train', requires_name=False, point_num=1, mask_num=args.mask_num)
    train_dataset3 = FivesDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = EyesDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = StareDataset('data/stare', image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = TrainingDataset('data_demo', image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    #train_dataset = ConcatDataset([train_dataset1,train_dataset2])
    train_loader = DataLoader(train_dataset3, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataset = FivesDataset(args.data_path, image_size=args.image_size, mode='test', point_num=1, mask_num=args.mask_num, requires_name = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    print('*******Train data:', len(train_dataset3))
    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    # lora_sam.sam = DataParallel(lora_sam.sam,device_ids=[1,2,3,0])
    best_loss = 1e10
    l = len(train_loader)
    
    for epoch in range(0, args.epochs):
        lora_sam.sam.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, lora_sam.sam, optimizer, train_loader, epoch, criterion)
        val_losses = val_one_epoch(args, lora_sam.sam, optimizer, test_loader, epoch, criterion)
        if args.lr_scheduler is not None:
            scheduler.step()
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics},Val loss:{val_losses:.4f}")
        save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
        state = {'model': lora_sam.sam.float().state_dict(), 'optimizer': optimizer}
        torch.save(state, save_path)

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))