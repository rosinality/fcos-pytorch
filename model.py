import math

import torch
from torch import nn
from torch.nn import functional as F

from loss import FCOSLoss
from postprocess import FCOSPostprocessor


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, top_blocks=None):
        super().__init__()

        self.inner_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i, in_channel in enumerate(in_channels, 1):
            if in_channel == 0:
                self.inner_convs.append(None)
                self.out_convs.append(None)

                continue

            inner_conv = nn.Conv2d(in_channel, out_channel, 1)
            feat_conv = nn.Conv2d(out_channel, out_channel, 3, padding=1)

            self.inner_convs.append(inner_conv)
            self.out_convs.append(feat_conv)

        self.apply(init_conv_kaiming)

        self.top_blocks = top_blocks

    def forward(self, inputs):
        inner = self.inner_convs[-1](inputs[-1])
        outs = [self.out_convs[-1](inner)]

        for feat, inner_conv, out_conv in zip(
            inputs[:-1][::-1], self.inner_convs[:-1][::-1], self.out_convs[:-1][::-1]
        ):
            if inner_conv is None:
                continue

            upsample = F.interpolate(inner, scale_factor=2, mode='nearest')
            inner_feat = inner_conv(feat)
            inner = inner_feat + upsample
            outs.insert(0, out_conv(inner))

        if self.top_blocks is not None:
            top_outs = self.top_blocks(outs[-1], inputs[-1])
            outs.extend(top_outs)

        return outs


class FPNTopP6P7(nn.Module):
    def __init__(self, in_channel, out_channel, use_p5=True):
        super().__init__()

        self.p6 = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

        self.apply(init_conv_kaiming)

        self.use_p5 = use_p5

    def forward(self, f5, p5):
        input = p5 if self.use_p5 else f5

        p6 = self.p6(input)
        p7 = self.p7(F.relu(p6))

        return p6, p7


class FCOSHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior):
        super().__init__()

        n_class = n_class - 1

        cls_tower = []
        bbox_tower = []

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, input):
        logits = []
        bboxes = []
        centers = []

        for feat, scale in zip(input, self.scales):
            cls_out = self.cls_tower(feat)

            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))

            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))

            bboxes.append(bbox_out)

        return logits, bboxes, centers


class FCOS(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()

        self.backbone = backbone
        fpn_top = FPNTopP6P7(
            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
        )
        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)
        self.head = FCOSHead(
            config.out_channel, config.n_class, config.n_conv, config.prior
        )
        self.postprocessor = FCOSPostprocessor(
            config.threshold,
            config.top_n,
            config.nms_threshold,
            config.post_top_n,
            config.min_size,
            config.n_class,
        )
        self.loss = FCOSLoss(
            config.sizes,
            config.gamma,
            config.alpha,
            config.iou_loss_type,
            config.center_sample,
            config.fpn_strides,
            config.pos_radius,
        )

        self.fpn_strides = config.fpn_strides

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)

    def forward(self, input, image_sizes=None, targets=None):
        features = self.backbone(input)
        features = self.fpn(features)
        cls_pred, box_pred, center_pred = self.head(features)
        # print(cls_pred, box_pred, center_pred)
        location = self.compute_location(features)

        if self.training:
            loss_cls, loss_box, loss_center = self.loss(
                location, cls_pred, box_pred, center_pred, targets
            )
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
            }

            return None, losses

        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )

            return boxes, None

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location
