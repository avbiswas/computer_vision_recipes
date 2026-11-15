"""
Step 3 - TennisNet-lite: a small encoder-decoder that outputs heatmaps.

ENCODER  the first 3 stages of an ImageNet-pretrained ResNet-18 (we drop layer4:
         it is 8M of ResNet-18's 11M weights and we don't need its deep semantics).
         3 grayscale frames = 3 channels, so the pretrained first conv works unchanged.
         color=True: 3 colour frames = 9 channels. We copy the pretrained R,G,B filters
         to each frame (divided by 3), so at the start it "sees" the average of the 3 frames.

DECODER  upsample back to 1/2 resolution, mixing in the sharper early features
         ("skip connections", like a U-Net).

HEADS    one small conv that outputs 18 maps:
           0      ball heatmap
           1-14   court keypoint heatmaps
           15     player-centre heatmap
           16-17  player width, height
"""
import torch
import torch.nn as nn
import torchvision

NUM_HEATMAPS = 16


def conv_bn_relu(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Up(nn.Module):
    """Upsample the deep features 2x, concatenate the skip features, mix with a 3x3 conv."""

    def __init__(self, c_deep, c_skip, c_out):
        super().__init__()
        self.mix = conv_bn_relu(c_deep + c_skip, c_out)

    def forward(self, deep, skip):
        deep = nn.functional.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.mix(torch.cat([deep, skip], dim=1))


class TennisNetLite(nn.Module):
    def __init__(self, pretrained=True, color=False):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        if color:
            rgb_filters = r.conv1.weight.data                    # (64, 3, 7, 7)
            r.conv1 = nn.Conv2d(9, 64, 7, stride=2, padding=3, bias=False)
            r.conv1.weight.data = torch.cat([rgb_filters] * 3, dim=1) / 3
        # encoder                                               resolution   channels
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # 1/2          64
        self.stage1 = nn.Sequential(r.maxpool, r.layer1)       # 1/4          64
        self.stage2 = r.layer2                                 # 1/8          128
        self.stage3 = r.layer3                                 # 1/16         256
        # decoder
        self.up3 = Up(256, 128, 128)                           # 1/16 -> 1/8
        self.up2 = Up(128, 64, 64)                             # 1/8  -> 1/4
        self.up1 = Up(64, 64, 64)                              # 1/4  -> 1/2
        # heads
        self.head = nn.Sequential(conv_bn_relu(64, 64), nn.Conv2d(64, NUM_HEATMAPS + 2, 1))
        nn.init.constant_(self.head[-1].bias[:NUM_HEATMAPS], -4.0)   # start with "nothing anywhere"

    def forward(self, x):
        x = (x - 0.45) / 0.225                                 # ImageNet-style normalisation
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        d = self.up3(s3, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        out = self.head(d)
        heatmaps = torch.sigmoid(out[:, :NUM_HEATMAPS])        # values in 0..1
        sizes = out[:, NUM_HEATMAPS:]                          # player width, height
        return heatmaps, sizes


if __name__ == "__main__":
    model = TennisNetLite(color=True)
    params = sum(p.numel() for p in model.parameters())
    heatmaps, sizes = model(torch.zeros(1, 9, 288, 512))
    print(f"{params / 1e6:.2f}M parameters | heatmaps {tuple(heatmaps.shape)} | sizes {tuple(sizes.shape)}")
