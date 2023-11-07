from functools import partial
import numpy as np

import torch
from torch import nn

from Models import pvt_v2
from timm.models.vision_transformer import _cfg
from ops_dcnv3.modules import DCNv3


def run_operator(conv_x, conv_y, input, operator_type="sobel"):
    if operator_type == "sobel" or "prewitt" or "scharr":
        g_x = conv_x(input)
        g_y = conv_y(input)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
        return torch.sigmoid(g) * input
    elif operator_type == "Laplacian" or "LOG":
        g = conv_x(input)
        return torch.sigmoid(g) * input

def get_operator(in_chan, out_chan, operator_type="sobel"):
    if operator_type == "sobel":
        filter_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]).astype(np.float32)
        filter_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]).astype(np.float32)
    elif operator_type == "prewitt":
        filter_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]).astype(np.float32)
        filter_y = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ]).astype(np.float32)
    elif operator_type == "scharr":
        filter_x = np.array([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3],
        ]).astype(np.float32)
        filter_y = np.array([
            [-3, 10, 3],
            [0, 0, 0],
            [3, 10, 3],
        ]).astype(np.float32)
    elif operator_type == "Laplacian":
        filter_x = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ]).astype(np.float32)
    elif operator_type == "LOG":
        filter_x = np.array([
            [0, 0, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 0, 0],
        ]).astype(np.float32)

    if operator_type == "scharr":
        filter_x = filter_x.reshape((1, 1, 3, 3))
        filter_x = np.repeat(filter_x, in_chan, axis=1)
        filter_x = np.repeat(filter_x, out_chan, axis=0)

        filter_y = filter_y.reshape((1, 1, 3, 3))
        filter_y = np.repeat(filter_y, in_chan, axis=1)
        filter_y = np.repeat(filter_y, out_chan, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_y = torch.from_numpy(filter_y)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        filter_y = nn.Parameter(filter_y, requires_grad=False)
        conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y.weight = filter_y
        operator_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
        operator_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
        return operator_x, operator_y
    if operator_type == "Laplacian":
        filter_x = filter_x.reshape((1, 1, 3, 3))
        filter_x = np.repeat(filter_x, in_chan, axis=1)
        filter_x = np.repeat(filter_x, out_chan, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        operator_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
        return operator_x
    if operator_type == "LOG":
        filter_x = filter_x.reshape((1, 1, 5, 5))
        filter_x = np.repeat(filter_x, in_chan, axis=1)
        filter_x = np.repeat(filter_x, out_chan, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=5, stride=1, padding=2, bias=False)
        conv_x.weight = filter_x
        operator_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
        return operator_x


class Operator_Block(nn.Module):
    def __init__(self, in_channels, out_channels, operator_first_type="scharr", operator_second_type="LOG"):
        super(Operator_Block, self).__init__()
        self.operator1, self.operator2 = get_operator(in_channels, in_channels, operator_type=operator_first_type)
        self.operator3 = get_operator(in_channels, in_channels, operator_type=operator_second_type)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.SiLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x = run_operator(self.operator1, self.operator2, x, operator_type="scharr")
        y = run_operator(self.operator3, self.operator3, x, operator_type="LOG")
        h = x + y
        out = self.block(h)

        return out


class StemLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, out_channels // 2)
        # self.norm = nn.BatchNorm2d(out_channels//2)
        self.act = nn.SiLU()
        # self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class DCNv3_Block(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=1, g=1, d=1):
        super().__init__()
        self.block_in = DCNv3(in_channels, kernel_size=k, stride=s, group=g, pad=p, dilation=d)
        # self.block_in = Operator_Block(in_channels, operator_type = "prewitt")

        self.block_out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            # nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            # nn.BatchNorm2d(in_channels),
            nn.SiLU()
            # nn.ReLU()
        )

        self.skip = nn.Identity()

        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        ### DCNv3
        x = x.permute(0, 2, 3, 1)
        x = self.block_in(x)
        x = x.permute(0, 3, 1, 2)

        # #### operator
        # x = self.block_in(x)

        # x = x + self.skip(x)
        x = self.block_out(x)
        # x = x + self.skip(x)
        x = self.down_sample(x)

        return x


# class DCNv3_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, k=1, s=1, p=1, g=1, d=1):

#         super().__init__()
#         self.block_in1 = DCNv3(in_channels, kernel_size=k, stride=s, group=g, pad=p, dilation=d)
#         self.bolck_in12 = nn.Sequential(
#             nn.GroupNorm(32, in_channels),
#             # nn.BatchNorm2d(in_channels),
#             nn.SiLU(),
#             # nn.ReLU(),
#         )
#         self.block_in2 = Operator_Block(in_channels, operator_type="sobel")

#         self.block_out = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(32, in_channels),
#             # nn.BatchNorm2d(in_channels),
#             nn.SiLU()
#             # nn.ReLU()
#         )

#         self.skip = nn.Identity()

#         self.down_sample = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.GroupNorm(32, out_channels),
#             # nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x):
#         ####DCNv3
#         x1 = x.permute(0, 2, 3, 1)
#         x1= self.block_in1(x1)
#         x1 = x1.permute(0, 3, 1, 2)
#         x1 = self.bolck_in12(x1)
#         #####Sobel
#         x2 = self.block_in2(x)

#         x12 = x1 + x2
#         x_out = self.block_out(x12)
#         x = x_out + self.skip(x)
#         x = self.down_sample(x)

#         return x


class Encoder(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("./pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.stem = StemLayer(in_channels=3, out_channels=64)
        self.dcnv3_1 = DCNv3_Block(in_channels=32, out_channels=64, k=3, s=1, p=1, g=4, d=1)
        self.dcnv3_2 = DCNv3_Block(in_channels=64, out_channels=128, k=3, s=1, p=1, g=8, d=1)
        self.dcnv3_3 = DCNv3_Block(in_channels=128, out_channels=320, k=3, s=1, p=1, g=16, d=1)
        self.dcnv3_4 = DCNv3_Block(in_channels=320, out_channels=512, k=3, s=1, p=1, g=32, d=1)

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        transformer_pyramid = self.get_pyramid(x)
        convolution_pyramid = []
        x_stem = self.stem(x)
        x1 = self.dcnv3_1(x_stem)
        convolution_pyramid.append(x1)
        x2 = self.dcnv3_2(x1)
        convolution_pyramid.append(x2)
        x3 = self.dcnv3_3(x2)
        convolution_pyramid.append(x3)
        x4 = self.dcnv3_4(x3)
        convolution_pyramid.append(x4)

        return transformer_pyramid, convolution_pyramid


# class SemanticInteractionMoudle(nn.Module):
#     # Same out_channels(64)
#     def __init__(self, in_channels, out_channels):

#         super().__init__()

#         self.local_embedding = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.GroupNorm(32, out_channels)
#             # nn.BatchNorm2d(out_channels)
#         )

#         self.local_global_interaction =nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.GroupNorm(32, out_channels),
#             # nn.BatchNorm2d(out_channels),
#             nn.Sigmoid()
#         )

#         self.global_embedding = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.GroupNorm(32, out_channels)
#             # nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x_l, x_g):
#         local_feat = self.local_embedding(x_l)
#         global_act = self.local_global_interaction(x_g)
#         global_feat = self.global_embedding(x_g)
#         out = local_feat * global_act + global_feat

#         return out

class EdgeDetectionGuidedMoudle(nn.Module):
    # Same out_channels(64)
    def __init__(self, in_channels, out_channels, next_add=False):
        self.next_add = next_add

        super().__init__()

        self.local_embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels)
            nn.SiLU()
        )

        self.global_embedding =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.concat = nn.Sequential(
            Operator_Block(out_channels * 2, out_channels, operator_first_type="scharr", operator_second_type="LOG")
        )

        if self.next_add:
            self.context_embedding = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, out_channels),
                # nn.BatchNorm2d(out_channels)
                nn.SiLU()
            )
        else:
            self.context_embedding = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, out_channels),
                # nn.BatchNorm2d(out_channels)
                nn.SiLU()
            )

        self.upsmaple = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x_l, x_g, x_n):
        local_feat = self.local_embedding(x_l)
        global_feat = self.global_embedding(x_g)
        context_feat = self.context_embedding(x_n)

        out0 = torch.concat((local_feat, global_feat), dim=1)
        out0 = self.concat(out0)
        out1 = out0 + context_feat

        out2 = self.upsmaple(out1)

        return out1, out2

class SemanticInteractionMoudle(nn.Module):
    # Same out_channels(64)
    def __init__(self, in_channels, out_channels, next_add=False):
        self.next_add = next_add

        super().__init__()

        self.local_embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels)
            nn.SiLU()
        )

        self.local_global_interaction =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # self.local_global_interaction = nn.Sequential(
        #     Operator_Block(in_channels, operator_type="sobel"),
        # )
        # self.concat = nn.Sequential(
        #     nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(32, out_channels),
        #     # nn.BatchNorm2d(out_channels),
        # )

        if self.next_add:
            self.global_embedding = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, out_channels),
                # nn.BatchNorm2d(out_channels)
                nn.SiLU()
            )
        else:
            self.global_embedding = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, out_channels),
                # nn.BatchNorm2d(out_channels)
                nn.SiLU()
            )

        self.upsmaple = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x_l, x_g, x_n):
        local_feat = self.local_embedding(x_l)
        global_act = self.local_global_interaction(x_g)
        global_feat = self.global_embedding(x_n)

        # out0 = torch.concat((local_feat, global_act), dim=1)
        # out0 = self.concat(out0)
        # out1 = out0 + global_feat

        out1 = local_feat * global_act + global_feat
        out2 = self.upsmaple(out1)

        return out1, out2


class Seg_head(nn.Module):
    def __init__(self, num_class=1, size=352):
        super().__init__()

        # self.upsample1 = nn.Upsample(size=88, mode='bilinear')
        # self.upsample2 = nn.Upsample(size=size, mode='bilinear')
        self.upsample1 = nn.Upsample(size=88, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        ### +
        self.PH = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, num_channels=64),
            # nn.BatchNorm2d(64),
            nn.SiLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, stride=1)
        )
        # ### concat
        # self.PH = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
        #     # nn.GroupNorm(32, num_channels=64),
        #     nn.BatchNorm2d(64),
        #     # nn.SiLU(),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, stride=1)
        # )

    def forward(self, x):
        # ### +
        # pred = x[0]   # 64, 88, 88
        # for i in range(1, 4):
        #     pred += self.upsample1(x[i])

        ### + and depp supervision
        pred = self.upsample1(x[0])  # 64, 88, 88
        for i in range(1, len(x)):
            pred += self.upsample1(x[i])

        # ### concat
        # pred = x[0]  # 64, 88, 88
        # for i in range(1, 4):
        #     pred = torch.concat([pred, self.upsample1(x[i])], dim=1)

        x = self.upsample2(pred)  # 88,88,352
        x = self.PH(x)

        return x


# class Net(nn.Module):
#     def __init__(self, num_class=1, size=352):

#         super().__init__()
#         self.encoder = Encoder()
#         self.SIM1 = SemanticInteractionMoudle(in_channels=64, out_channels=64)
#         self.SIM2 = SemanticInteractionMoudle(in_channels=128, out_channels=64)
#         self.SIM3 = SemanticInteractionMoudle(in_channels=320, out_channels=64)
#         self.SIM4 = SemanticInteractionMoudle(in_channels=512, out_channels=64)
#         self.head = Seg_head(num_class, size)

#     def forward(self, x):
#         sim = []
#         transformer_pyramid, convolution_pyramid = self.encoder(x)
#         x1 = self.SIM1(transformer_pyramid[0],convolution_pyramid[0])
#         sim.append(x1)
#         x2 = self.SIM2(transformer_pyramid[1],convolution_pyramid[1])
#         sim.append(x2)
#         x3 = self.SIM3(transformer_pyramid[2],convolution_pyramid[2])
#         sim.append(x3)
#         x4 = self.SIM4(transformer_pyramid[3],convolution_pyramid[3])
#         sim.append(x4)
#         out = self.head(sim)

#         return out

class Net(nn.Module):
    def __init__(self, num_class=1, size=352):
        super().__init__()
        self.encoder = Encoder()
        self.EDGM1 = EdgeDetectionGuidedMoudle(in_channels=64, out_channels=64, next_add=True)
        self.EDGM2 = EdgeDetectionGuidedMoudle(in_channels=128, out_channels=64, next_add=True)
        self.SIM3 = SemanticInteractionMoudle(in_channels=320, out_channels=64, next_add=True)
        self.SIM4 = SemanticInteractionMoudle(in_channels=512, out_channels=64, next_add=False)
        self.head = Seg_head(num_class, size)

    def forward(self, x):
        sim = []
        transformer_pyramid, convolution_pyramid = self.encoder(x)
        x4, x44 = self.SIM4(transformer_pyramid[3], convolution_pyramid[3], transformer_pyramid[3])
        x3, x33 = self.SIM3(transformer_pyramid[2], convolution_pyramid[2], x44)
        x2, x22 = self.EDGM2(transformer_pyramid[1], convolution_pyramid[1], x33)
        x1, x11 = self.EDGM1(transformer_pyramid[0], convolution_pyramid[0], x22)
        sim.append(x1)
        sim.append(x2)
        sim.append(x3)
        sim.append(x4)
        out = self.head(sim)
        out_a1 = self.head(sim[1:4])
        out_a2 = self.head(sim[2:4])
        out_a3 = self.head(sim[3:4])

        # return out
        return out, out_a1, out_a2, out_a3

