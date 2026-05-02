import math
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class DecoupleLayer(nn.Module):
    def __init__(self, in_c=2048, out_c=64):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc

class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)
        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

class CDFAPreprocess(nn.Module):
    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x

class Baseline3Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 骨干网络（Encoder）：使用标准的 ResNet-50，加载官方 ImageNet 预训练权重
        backbone = resnet50(pretrained=True)
        
        # 严格保持原命名前缀，确保 Stage1 的参数可以完美 load 到这里
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) # [B, 64, H/2, W/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)           # Block 1 (f1): [B, 256, H/4, W/4]
        self.layer2 = backbone.layer2                                            # Block 2 (f2): [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3                                            # Block 3 (f3): [B, 1024, H/16, W/16]
        self.layer4 = backbone.layer4                                            # Block 4 (f4): [B, 2048, H/32, W/32]
        
        fuse_dim = 64
        self.conv_f1 = CBR(256, fuse_dim)
        self.conv_f2 = CBR(512, fuse_dim)
        self.conv_f3 = CBR(1024, fuse_dim)
        self.conv_f4 = CBR(2048, fuse_dim)
        
        # 2. 引入 SID 物理结构: 接收 layer4 的 2048 维度输出
        # 我们这里不再有 AuxiliaryHead，所以废弃该头，后续不输出辅助 mask
        self.decouple_layer = DecoupleLayer(2048, fuse_dim)
        
        # 3. 引入 CDFA 特征聚合逻辑
        # f_fg 和 f_bg 从 f4 分离出来，分辨率在 1/32
        # 我们需要将其上采样匹配 f4(1/32), f3(1/16), f2(1/8), f1(1/4)
        self.preprocess_fg4 = CDFAPreprocess(fuse_dim, fuse_dim, 1)
        self.preprocess_bg4 = CDFAPreprocess(fuse_dim, fuse_dim, 1)
        
        self.preprocess_fg3 = CDFAPreprocess(fuse_dim, fuse_dim, 2)
        self.preprocess_bg3 = CDFAPreprocess(fuse_dim, fuse_dim, 2)
        
        self.preprocess_fg2 = CDFAPreprocess(fuse_dim, fuse_dim, 4)
        self.preprocess_bg2 = CDFAPreprocess(fuse_dim, fuse_dim, 4)
        
        self.preprocess_fg1 = CDFAPreprocess(fuse_dim, fuse_dim, 8)
        self.preprocess_bg1 = CDFAPreprocess(fuse_dim, fuse_dim, 8)
        
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        drop_rate = 0.1
        self.cdfa4 = ContrastDrivenFeatureAggregation(fuse_dim, fuse_dim, 4, attn_drop=drop_rate, proj_drop=drop_rate)
        self.cdfa3 = ContrastDrivenFeatureAggregation(fuse_dim + fuse_dim, fuse_dim, 4, attn_drop=drop_rate, proj_drop=drop_rate)
        self.cdfa2 = ContrastDrivenFeatureAggregation(fuse_dim + fuse_dim, fuse_dim, 4, attn_drop=drop_rate, proj_drop=drop_rate)
        self.cdfa1 = ContrastDrivenFeatureAggregation(fuse_dim + fuse_dim, fuse_dim, 4, attn_drop=drop_rate, proj_drop=drop_rate)

        # 4. Decoder 保持单尺度 (无并行 SA-Decoder)
        # CDFA 生成了逐层向上融合的特征 c1, c2, c3, c4
        # 我们将其与纯净基线保持一致地统一上采样和 Concat
        self.up_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        
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
        f1 = self.layer1(x0) # Block 1 (H/4)
        f2 = self.layer2(f1) # Block 2 (H/8)
        f3 = self.layer3(f2) # Block 3 (H/16)
        f4 = self.layer4(f3) # Block 4 (H/32)
        
        # 2. 3x3 Conv normalization (生成图中的 e1, e2, e3, e4)
        e1 = self.conv_f1(f1)
        e2 = self.conv_f2(f2)
        e3 = self.conv_f3(f3)
        e4 = self.conv_f4(f4)
        
        # 3. SID: 解耦提取 f_fg, f_bg
        f_fg, f_bg, f_uc = self.decouple_layer(f4)
        # 注意: 这里绝不生成和返回任何 auxiliary masks (M_fg, M_bg)
        
        # 4. CDFA 特征聚合：用于指导 f1, f2, f3, f4 的逐层向上融合
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)
        
        c4 = self.cdfa4(e4, f_fg4, f_bg4)
        c4_up = self.up2X(c4)
        
        e3_c4 = torch.cat([e3, c4_up], dim=1)
        c3 = self.cdfa3(e3_c4, f_fg3, f_bg3)
        c3_up = self.up2X(c3)
        
        e2_c3 = torch.cat([e2, c3_up], dim=1)
        c2 = self.cdfa2(e2_c3, f_fg2, f_bg2)
        c2_up = self.up2X(c2)
        
        e1_c2 = torch.cat([e1, c2_up], dim=1)
        c1 = self.cdfa1(e1_c2, f_fg1, f_bg1)
        
        # 5. Parallel Upsampling
        c2_up_out = self.up_2x(c2)
        c3_up_out = self.up_4x(c3)
        c4_up_out = self.up_8x(c4)
        
        # 6. Concat (在 H/4 的分辨率下一次性拼接) 依然使用融合特征
        fused = torch.cat([c1, c2_up_out, c3_up_out, c4_up_out], dim=1)
        
        # 7. Decoder (输出唯一的预测)
        mask_pred = self.decoder(fused)
        
        # 严格遵守：仅返回唯一预测掩码
        return mask_pred
