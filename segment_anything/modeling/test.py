import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)

        q = self.query(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = self.key(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
        v = self.value(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class attentionBlock(nn.Module):
    def __init__(self):
        super(attentionBlock, self).__init__()
        self.small = nn.Sequential(
            nn.Conv2d(3,64,1,1,0),
            nn.Conv2d(64,32,3,1,2,1),
            nn.Conv2d(32,16,3,1,0,1),
            nn.GELU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(3,64,1,1,0),
            nn.Conv2d(64,32,3,1,3,2),
            nn.Conv2d(32,16,3,1,0,1),
            nn.GELU()
        )
        self.big = nn.Sequential(
            nn.Conv2d(3,64,1,1,0),
            nn.Conv2d(64,32,3,1,4,3),
            nn.Conv2d(32,16,3,1,0,1),
            nn.GELU()
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm([48,256,256]),
            nn.Conv2d(48,16,3,1,0),
            nn.Conv2d(16,2,5,3,0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        small = self.small(x)
        middle = self.middle(x)
        big = self.big(x)
        out = torch.cat([small,middle,big],dim=1)
        out = self.mlp(out)
        out = self.avgpool(out)
        out = F.softmax(out,dim=1)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = attentionBlock()
    out = model(x)
    print(out.shape)