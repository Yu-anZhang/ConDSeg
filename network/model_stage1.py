import os
import torch
import torch.nn as nn

from network.resnet import resnet50, resnet101
from network_pvt.pvtv2 import pvt_v2_b2


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


class adjust(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4, out_c):
        super().__init__()
        self.conv1 = CBR(in_c1, 64, kernel_size=1, padding=0, act=True)
        self.conv2 = CBR(in_c2, 64, kernel_size=1, padding=0, act=True)
        self.conv3 = CBR(in_c3, 64, kernel_size=1, padding=0, act=True)
        self.conv4 = CBR(in_c4, 64, kernel_size=1, padding=0, act=True)
        self.conv_fuse = nn.Conv2d(4 * 64, out_c, kernel_size=1, padding=0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        x = self.sig(x)
        return x


class ConDSegStage1(nn.Module):
    def __init__(self, H=256, W=256, backbone_name="resnet50", pvt_pretrained_path=None):
        super().__init__()

        self.H = H
        self.W = W
        self.backbone_name = backbone_name.lower()
        self.backbone = None

        if self.backbone_name == "resnet50":
            backbone = resnet50()
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            head_channels = (64, 256, 512, 1024)
            up_scales = (2, 4, 8, 16)
        elif self.backbone_name in {"resnet101", "resnet100"}:
            backbone = resnet101(pretrained=True)
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            head_channels = (64, 256, 512, 1024)
            up_scales = (2, 4, 8, 16)
        elif self.backbone_name in {"pvt", "pvtv2", "pvt_v2_b2"}:
            self.backbone = pvt_v2_b2()
            self.layer0 = None
            self.layer1 = None
            self.layer2 = None
            self.layer3 = None
            self._load_pvt_weights(pvt_pretrained_path)
            head_channels = (64, 128, 320, 512)
            up_scales = (4, 8, 16, 32)
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Choose from: resnet50, resnet101, resnet100, pvt_v2_b2."
            )

        self.up_2x2 = nn.Upsample(scale_factor=up_scales[0], mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=up_scales[1], mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=up_scales[2], mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=up_scales[3], mode="bilinear", align_corners=True)

        self.head = adjust(*head_channels, 1)

    def _load_pvt_weights(self, pvt_pretrained_path=None):
        candidate_paths = []
        if pvt_pretrained_path:
            candidate_paths.append(pvt_pretrained_path)
        candidate_paths.append("./pretrained_pth/pvt_v2_b2.pth")

        for path in candidate_paths:
            if path and os.path.exists(path):
                save_model = torch.load(path, map_location="cpu")
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                return

    def forward(self, image):
        if self.backbone is None:
            x1 = self.layer0(image)
            x2 = self.layer1(x1)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
        else:
            x1, x2, x3, x4 = self.backbone(image)

        x1 = self.up_2x2(x1)
        x2 = self.up_4x4(x2)
        x3 = self.up_8x8(x3)
        x4 = self.up_16x16(x4)

        pred = self.head(x1, x2, x3, x4)
        return pred


if __name__ == "__main__":
    model = ConDSegStage1(backbone_name="resnet50").cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)
