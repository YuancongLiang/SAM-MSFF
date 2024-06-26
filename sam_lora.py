# Sheng Wang at Apr 6 2023
# What a time to be alive (first half of 2023)

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from torchsummary import summary
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        q=True,
        k=True,
        v=True,
    ):
        super().__init__()
        self.qkv = qkv
        self.q = q
        self.k = k
        self.v = v
        if self.q:
            self.linear_a_q = linear_a_q
            self.linear_b_q = linear_b_q
        if self.k:
            self.linear_a_k = linear_a_k
            self.linear_b_k = linear_b_k
        if self.v:
            self.linear_a_v = linear_a_v
            self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        if self.q:
            new_q = self.linear_b_q(self.linear_a_q(x))
            qkv[:, :, :, : self.dim] += new_q
        if self.k:
            new_k = self.linear_b_k(self.linear_a_k(x))
            qkv[:, :, :, self.dim:-self.dim] += new_k
        if self.v:
            new_v = self.linear_b_v(self.linear_a_v(x))
            qkv[:, :, :, -self.dim :] += new_v
        return qkv

class _LoRA_output(nn.Module):

    def __init__(
        self,
        proj: nn.Module,
        linear_a_proj: nn.Module,
        linear_b_proj: nn.Module,
    ):
        super().__init__()
        self.proj = proj
        self.linear_a_proj = linear_a_proj
        self.linear_b_proj = linear_b_proj
        self.dim = proj.in_features
        self.in_features = proj.in_features
        self.out_features = proj.out_features
        self.w_identity = torch.eye(proj.in_features)

    def forward(self, x):
        proj = self.proj(x)  
        new_proj = self.linear_b_proj(self.linear_a_proj(x))
        proj += new_proj
        return proj

class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, q=True,k=True,v=True,out=True,lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            w_proj_linear = blk.attn.proj
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            w_a_linear_proj = nn.Linear(self.dim, r, bias=False)
            w_b_linear_proj = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_As.append(w_a_linear_proj)
            self.w_Bs.append(w_b_linear_proj)
            blk.attn.qkv = _LoRA_qkv(
                qkv=w_qkv_linear,
                linear_a_q=w_a_linear_q,
                linear_b_q=w_b_linear_q,
                linear_a_k=w_a_linear_k,
                linear_b_k=w_b_linear_k,
                linear_a_v=w_a_linear_v,
                linear_b_v=w_b_linear_v,
                q=q,
                k=k,
                v=v,
            )
            if out:
                blk.attn.proj = _LoRA_output(
                    proj=w_proj_linear,
                    linear_a_proj=w_a_linear_proj,
                    linear_b_proj=w_b_linear_proj,
                )
        self.reset_parameters()
        self.sam = sam_model


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            #nn.init.normal_(w_A.weight)
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)


if __name__ == "__main__":
    args = argparse.Namespace()
    args.image_size = 256
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.encoder_adapter = True
    args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"
    # args.sam_checkpoint = "workdir/models/sam-med2d/epoch30_sam.pth"
    model = sam_model_registry["vit_b"](args).to("cpu")
    lora_sam = LoRA_Sam(model,64).to("cpu")
    summary(lora_sam.sam.image_encoder, (3, 256, 256), device="cpu")
    # with open(args.sam_checkpoint, "rb") as f:
    #         state_dict = torch.load(f)
    #         lora_sam.sam.load_state_dict(state_dict['model'])
    # print(lora_sam.sam)
    # for n, value in lora_sam.sam.named_parameters():
    #     print(n,value.requires_grad)
    
    # lora_sam.sam.image_encoder(torch.rand(size=(1,3,256,256)).to(device))