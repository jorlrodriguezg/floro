import torch
import torch.nn as nn

# _make_scratch modified with prefix support for multimodal feature extraction
def _make_scratch(in_shape, out_shape, groups=1, expand=False, prefix=""):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    setattr(scratch, f"{prefix}layer1_rn", nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups))
    setattr(scratch, f"{prefix}layer2_rn", nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups))
    setattr(scratch, f"{prefix}layer3_rn", nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups))
    if len(in_shape) >= 4:
        setattr(scratch, f"{prefix}layer4_rn", nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups))

    return scratch

# Residual Conv Unit with batch norm option
def get_norm_layer(features, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(features)
    elif norm_type == "gn":
        return nn.GroupNorm(num_groups=8, num_channels=features)
    else:
        return nn.Identity()

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, norm_type="bn"):
        super().__init__()
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.bn1 = get_norm_layer(features, norm_type)
        self.bn2 = get_norm_layer(features, norm_type)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return self.skip_add.add(out, x)

# Feature Fusion Block with optional upsampling and size specification
class FeatureFusionBlock(nn.Module):
    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        norm_type="bn", 
        expand=False, 
        align_corners=True,
        size=None
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        self.size = size

        out_features = features if not expand else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, norm_type)
        self.resConfUnit2 = ResidualConvUnit(features, activation, norm_type)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        modifier = {"scale_factor": 2} if (size is None and self.size is None) else {"size": self.size if size is None else size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output
