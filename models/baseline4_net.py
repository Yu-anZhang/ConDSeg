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

class DecoupleLayer(nn.Module):
    def __init__(self, in_c=2048, out_c=64):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc

class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  # 1/32 -> 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc

class Baseline4Network(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        fuse_dim = 64
        self.conv_f1 = CBR(256, fuse_dim)
        self.conv_f2 = CBR(512, fuse_dim)
        self.conv_f3 = CBR(1024, fuse_dim)
        
        self.decouple_layer = DecoupleLayer(2048, fuse_dim)
        
        # 引入 AuxiliaryHead 计算创新损失 (与无损失版区分)
        self.aux_head = AuxiliaryHead(fuse_dim)
        
        self.fuse_sid = CBR(fuse_dim * 3, fuse_dim)
        
        self.up_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        
        # 无 SA-Decoder 的单输出头
        self.decoder = nn.Sequential(
            CBR(fuse_dim * 4, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.layer0(x)  
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        e1 = self.conv_f1(f1)
        e2 = self.conv_f2(f2)
        e3 = self.conv_f3(f3)
        
        # SID
        f_fg, f_bg, f_uc = self.decouple_layer(f4)
        
        # 辅助掩码预测
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)
        
        e4 = self.fuse_sid(torch.cat([f_fg, f_bg, f_uc], dim=1))
        
        e2_up = self.up_2x(e2)
        e3_up = self.up_4x(e3)
        e4_up = self.up_8x(e4)
        
        fused = torch.cat([e1, e2_up, e3_up, e4_up], dim=1)
        out = self.decoder(fused)
        
        return out, mask_fg, mask_bg, mask_uc
