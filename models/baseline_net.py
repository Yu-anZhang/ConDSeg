import torch
import torch.nn as nn
from torchvision.models import resnet50

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BaselineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 骨干网络（Encoder）：使用标准的 ResNet-50，加载官方 ImageNet 预训练权重
        backbone = resnet50(pretrained=True)
        
        # 严格保持原命名前缀，确保 Stage1 的参数可以完美 load 到这里
        # 注意：这里的 layer0 和 layer1 的定义，与 Stage1 中完全一致，防止权重加载失效
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) # [B, 64, H/2, W/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)           # Block 1 (f1): [B, 256, H/4, W/4]
        self.layer2 = backbone.layer2                                            # Block 2 (f2): [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3                                            # Block 3 (f3): [B, 1024, H/16, W/16]
        self.layer4 = backbone.layer4                                            # Block 4 (f4): [B, 2048, H/32, W/32]
        
        # -------------------------------------------------------------
        # 严格按照 Ablation Study Figure 12 结构重构
        # -------------------------------------------------------------
        
        # 每个分支经过一个 3x3 Conv 降维到统一维度 (例如 64 通道)
        fuse_dim = 64
        self.conv_f1 = CBR(256, fuse_dim)
        self.conv_f2 = CBR(512, fuse_dim)
        self.conv_f3 = CBR(1024, fuse_dim)
        self.conv_f4 = CBR(2048, fuse_dim)
        
        # 并行上采样到 H/4 分辨率
        self.up_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        
        # Decoder 结构 (处理 Concat 后的特征，将其放大回原始分辨率 H)
        # concat 后输入维度为 64 * 4 = 256
        self.decoder = nn.Sequential(
            CBR(fuse_dim * 4, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), # -> H/2
            CBR(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), # -> H
            CBR(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Backbone Feature Extraction
        x0 = self.layer0(x)  
        f1 = self.layer1(x0) # Block 1
        f2 = self.layer2(f1) # Block 2
        f3 = self.layer3(f2) # Block 3
        f4 = self.layer4(f3) # Block 4
        
        # 2. 3x3 Conv normalization (生成图中的 e1, e2, e3, e4)
        e1 = self.conv_f1(f1)
        e2 = self.conv_f2(f2)
        e3 = self.conv_f3(f3)
        e4 = self.conv_f4(f4)
        
        # 3. Parallel Upsampling
        e2_up = self.up_2x(e2)
        e3_up = self.up_4x(e3)
        e4_up = self.up_8x(e4)
        
        # 4. Concat (在 H/4 的分辨率下一次性拼接)
        fused = torch.cat([e1, e2_up, e3_up, e4_up], dim=1)
        
        # 5. Decoder
        out = self.decoder(fused)
        
        return out
