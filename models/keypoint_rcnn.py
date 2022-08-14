import torch.nn as nn

from einops import rearrange
from torchvision.ops import RoIAlign

class RPN(nn.Module):
    def __init__(self, in_ch, out_ch=512, k=3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k

    def forward(self, x):
        x = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=(1, 1))(x)

        score = nn.Conv2d(self.out_ch, 2 * self.k, kernel_size=1)(x)
        coordinates = nn.Conv2d(self.out_ch, 4 * self.k, kernel_size=1)(x)

        coordinates = rearrange(coordinates, "b (d k) h w -> b (h w k) d", d=4, k=self.k)
        return score, coordinates


class KeyPointHead(nn.Module):
    def __init__(self, in_ch, num_keypoints):
        super().__init__()
        self.in_ch = in_ch
        self.keypoints = num_keypoints

    def forward(self, x):
        x = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)
        x = nn.ConvTranspose2d(self.in_ch, self.in_ch, kernel_size=2, stride=2)(x)
        x = nn.ReLU()(x)
        x = nn.ConvTranspose2d(self.in_ch, self.keypoints, kernel_size=2, stride=2)(x)
        
        return x


class ClassHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

        modules = []
        for cin, cout in zip(self.filters[:-1], self.filters[1:]):
            modules.append(nn.Linear(cin, cout))
            modules.append(nn.LeakyReLU())
            
        modules.append(nn.Linear(self.filters[-1], 2))
        modules.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class BoxHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

        modules = []
        for cin, cout in zip(self.filters[:-1], self.filters[1:]):
            modules.append(nn.Linear(cin, cout))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(self.filters[-1], 4 * 2))
 
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class ClassBoxBackbone(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.cout = cout


class KeypointRCNN(nn.Module):
    def __init__(self, backbone, k=9, keypoints=22):
        super().__init__()
        self.k = k
        self.keypoints = keypoints
        self.backbone = backbone
        self.rpn = RPN(in_ch=1280, k=self.k)
        self.roi_align = RoIAlign(spatial_scale=1. / 32, output_size=(14, 14), sampling_ratio=2)
        self.class_head = ClassHead([16, 32, 64])
        self.box_head = BoxHead([16, 32, 64])
        self.keypoint_head = KeyPointHead(1280, 22)
        self.class_box_backbone = ClassBoxBackbone(16, 16)

    def forward(self, x):
        feats = self.backbone(x)
        scores, boxes = self.rpn(feats)
        aligned = self.roi_align(feats, list(boxes))

        keypoint_mask = self.keypoint_head(aligned)

        flat = nn.Flatten()(aligned)
        flat_feats = self.class_box_backbone(flat)
        box_offsets = self.box_head(flat_feats)
        class_scores = self.class_head(flat_feats)

        return keypoint_mask, box_offsets, class_scores

