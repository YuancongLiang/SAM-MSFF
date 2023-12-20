import torch
import torch.nn.functional as F
import numpy as np
import argparse
from sam_lora import LoRA_Sam
from segment_anything import sam_model_registry
import cv2
import os
import datetime
from utils import get_logger
from Dataset import EyesDataset, stack_dict_batched, FivesDataset, StareDataset, Chasedb1Dataset, FivesPPO
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
torch.manual_seed(3407)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="ppo_clip", help="run model name")
    parser.add_argument("--epochs", type=int, default=90, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/eyes_patch", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--lr", type=float, default=1e-8, help="learning rate")
    parser.add_argument("--resume", type=str, default='/home/liangyuancong/SAM-Med2D/pretrain_model/patch.pth', help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/pretrained_lora.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=True, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--workers", type=int, default=8, help="amount of workers")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--lmbda", type=float, default=0.95, help="lambda")
    parser.add_argument("--beta", type=float, default=0.01, help="beta")
    parser.add_argument("--limit", type=float, default=0.1, help="limit")
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
    def __init__(self):
        super().__init__()
    def forward(self, mask, ground_truth):
        intersection = torch.sum(mask * ground_truth)
        union = torch.sum(mask) + torch.sum(ground_truth) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou

class GAE(nn.Module):
    pass

class PPO:
    def __init__(self, action_model, value_model, reward_model, reference_model, actor_lr, critic_lr, gamma, lmbda, beta, limit, device):
        self.action_model = action_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.epochs = 5
        # PPO的四个模型
        # 只有action和value模型需要训练
        self.actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.action_model.parameters()), actor_lr)
        self.critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.value_model.parameters()), critic_lr)
        # 优化器导入完成
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.beta = beta
        self.limit = limit
        self.epsilon = 1e-2
    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)
    def update(self, state, ground_truth):
        #reference_mask, _ = self.reference_model(state)
        action_mask, _ = self.action_model(state)
        _, value = self.value_model(state)
        log_prob = torch.log(action_mask) # 这里为什么会有无穷大啊？
        #kl_divergence = F.kl_div(log_prob, reference_mask, reduction='sum')
        reward = self.reward_model(action_mask, ground_truth)
        old_action_mask = action_mask.detach()
        # 到这里为止，我们得到了state序列，action序列，reward序列，,value序列，现在要想办法算出advantage序列
        value_next = torch.cat((value[1:], torch.tensor([[0.]], device=self.device)), dim=0)
        td_target = reward + self.gamma * value_next
        td_delta = td_target - value
        advantage = self.compute_advantage(td_delta).mean()
        if advantage.mean() < self.limit/1.5:
            self.beta = self.beta * 2
        elif advantage.mean() > self.limit*1.5:
            self.beta = self.beta / 2
        else:
            pass
        for epoch in range(self.epochs):
            action_mask, _ = self.action_model(state)
            ratio = ((action_mask + 1e-7) / (old_action_mask + 1e-7)).mean()
            # 截断算法
            actor_loss = -torch.min((-ratio * advantage).squeeze(), torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage).mean()
            # KL散度惩罚
            # actor_loss = torch.mean((-ratio * advantage).squeeze())+ self.beta * kl_divergence
            _, value = self.value_model(state)
            critic_loss = torch.mean(F.mse_loss(value, td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            loggers.info(f"epoch: {epoch + 1}, critic_loss: {critic_loss:.4f}, actor loss: {actor_loss:.4f}")
def pad_and_crop(image, patch_height=256, patch_width=256):

    _, height, width = image.shape

    # 计算需要填充的像素数量
    pad_height = (patch_height - (height % patch_height))% patch_height
    pad_width = (patch_width - (width % patch_width))% patch_width

    # 进行填充操作
    padded_image = TF.pad(image, (pad_width // 2, pad_width - (pad_width // 2), pad_height // 2, pad_height - (pad_height // 2)))
    cropped_images = []
    for i in range(0, height + pad_height, 256):
        for j in range(0, width + pad_width, 256):
            cropped_images.append(padded_image[:, i:i+256, j:j+256])
    return cropped_images
        

if __name__ == '__main__':
    args = parse_args()
    sam_action = sam_model_registry[args.model_type](args)
    lora_sam_action = LoRA_Sam(sam_action,r = 16).to(args.device)

    sam_value = sam_model_registry[args.model_type](args)
    lora_sam_value = LoRA_Sam(sam_value,r = 16).to(args.device)

    # sam_reference = sam_model_registry[args.model_type](args)
    # lora_sam_reference = LoRA_Sam(sam_reference,r = 16).to(args.device)

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            lora_sam_action.sam.load_state_dict(checkpoint['model'])
            lora_sam_value.sam.load_state_dict(checkpoint['model'])
            #lora_sam_reference.sam.load_state_dict(checkpoint['model'])
            print(f"*******load {args.resume}")

    action_model = ActionValueModel(lora_sam_action.sam)
    action_model = DataParallel(action_model, device_ids=[0, 1])
    # 动作模型载入完成
    value_model = ActionValueModel(lora_sam_value.sam)
    value_model = DataParallel(value_model, device_ids=[0, 1])
    # 价值模型载入完成
    # lora_sam_reference.sam.eval()
    # reference_model = ActionValueModel(lora_sam_reference.sam)
    # reference_model = DataParallel(reference_model, device_ids=[1, 0])   
    reference_model = None
    # 参考模型载入完成
    reward_model = RewardModel()
    # 奖励模型载入完成
    PPO_trainer = PPO(action_model=action_model, value_model=value_model, reward_model=reward_model, 
                      reference_model=reference_model, actor_lr=args.lr, critic_lr=args.lr, 
                      gamma=args.gamma, lmbda=args.lmbda, beta=args.beta, limit=args.limit,device=args.device)
    train_dataset = FivesPPO("data/FIVES", image_size=256, mode='train', requires_name=False, point_num=1, mask_num=1)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
    for batched_input in tqdm(train_loader):
        image_list = batched_input["image"].to(args.device).squeeze(0).permute(0, 3, 1, 2)
        label_list = batched_input["label"].to(args.device).squeeze(0).permute(0, 3, 1, 2)

        # image = batched_input["image"][0]
        # label = batched_input["label"][0]
        # # 这里开始拆分patch
        # image_list = pad_and_crop(image)
        # label_list = pad_and_crop(label)
        # image_list = torch.tensor(np.array([image.cpu().detach().numpy() for image in image_list])).to(args.device)
        # label_list = torch.tensor(np.array([label.cpu().detach().numpy() for label in label_list])).to(args.device)
        PPO_trainer.update(image_list, label_list)
        loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))