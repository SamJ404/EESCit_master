# EESCit super-resolution reconstruction
# Originally by SAM J, Authorized by SAM J
# Spatialoperator and Channeloperator followed by
# 'Tianfang Zhang, et.al. CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Eficient Mobile Applicationns,arXiv:2408.03703v2'
# 2025.4.1

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
from models import register
from utils import make_coord
from mmcv.cnn.bricks.conv_module import build_norm_layer

class EchoEdgeModule(nn.Module):

    def __init__(self, dim_Ci, dim_Co):
        super().__init__()
        # Trainable Sobel Operator
        with torch.no_grad():
            self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_dia = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_antidia = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.wx = nn.Parameter(torch.randn(dim_Co, dim_Ci, 1, 1))
        self.wy = nn.Parameter(torch.randn(dim_Co, dim_Ci, 1, 1))
        self.wdia = nn.Parameter(torch.randn(dim_Co, dim_Ci, 1, 1))
        self.wantidia = nn.Parameter(torch.randn(dim_Co, dim_Ci, 1, 1))

        self.T_sobel_x = nn.Parameter(self.sobel_x * self.wx)
        self.T_sobel_y = nn.Parameter(self.sobel_y * self.wy)
        self.T_sobel_dia = nn.Parameter(self.sobel_dia * self.wdia)
        self.T_sobel_antidia = nn.Parameter(self.sobel_antidia * self.wantidia)
        ##

        # Echo Structure
        self.echo_start = nn.Conv2d(dim_Ci, dim_Ci, 3, 1, 1)

        self.echo1_0 = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo1_1 = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)

        self.echo2_0 = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo2_1 = nn.Conv2d(dim_Ci, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo2_2 = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)

        self.echo_x = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo_dia = nn.Conv2d(dim_Ci, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo_antidia = nn.Conv2d(dim_Ci, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)
        self.echo_y = nn.Conv2d(dim_Ci // 2, dim_Ci, 3, 1, 1, groups=dim_Ci // 2)

        self.act_x = nn.LeakyReLU()
        self.act_y = nn.LeakyReLU()
        self.act_dia = nn.LeakyReLU()
        self.act_antidia = nn.LeakyReLU()
        ##

        self.affine_xy = nn.Conv2d(dim_Co*2, dim_Co, kernel_size=3, stride=1, padding=1, groups=dim_Co)
        self.affine_dia = nn.Conv2d(dim_Co*2, dim_Co, kernel_size=3, stride=1, padding=1, groups=dim_Co)
        self.affine = nn.Conv2d(dim_Co*2, dim_Co, kernel_size=3, stride=1, padding=1, groups=dim_Co)
        _, self.norm = build_norm_layer(dict(type='BN', momentum=0.03, eps=0.001), dim_Co)
        self.act = nn.LeakyReLU()


    def forward(self, x):
        x_echo = self.echo_start(x)
        x_echo00, x_echo01 = torch.chunk(x_echo, chunks=2, dim=1)

        x_echo00 = self.echo1_0(x_echo00)
        x_echo01 = self.echo1_1(x_echo01)
        x_echo10, x_echo11 = torch.chunk(x_echo00, chunks=2, dim=1)
        x_echo12, x_echo13 = torch.chunk(x_echo01, chunks=2, dim=1)

        x_echo10 = self.echo2_0(x_echo10)
        x_echo11 = self.echo2_1(torch.cat([x_echo11, x_echo12], dim=1))
        x_echo12 = self.echo2_2(x_echo13)
        x_echo20, x_echo21 = torch.chunk(x_echo10, chunks=2, dim=1)
        x_echo22, x_echo23 = torch.chunk(x_echo11, chunks=2, dim=1)
        x_echo24, x_echo25 = torch.chunk(x_echo12, chunks=2, dim=1)

        x_horizontal = self.act_x(self.echo_x(x_echo20))
        x_diagnal = self.act_dia(self.echo_dia(torch.cat([x_echo21, x_echo22], dim=1)))
        x_antidiagnal = self.act_antidia(self.echo_antidia(torch.cat([x_echo23, x_echo24], dim=1)))
        x_vertical = self.act_y(self.echo_y(x_echo25))


        x_horizontal = F.conv2d(x_horizontal, self.T_sobel_x, padding=1)
        x_vertical = F.conv2d(x_vertical, self.T_sobel_y, padding=1)
        x_diagnal = F.conv2d(x_diagnal, self.T_sobel_dia, padding=1)
        x_antidiagnal = F.conv2d(x_antidiagnal, self.T_sobel_antidia, padding=1)

        x_xy = self.affine_xy(torch.cat([x_horizontal, x_vertical], 1))
        x_dia = self.affine_dia(torch.cat([x_diagnal, x_antidiagnal], 1))
        x_edge = self.affine(torch.cat([x_xy, x_dia], dim=1)) + x
        x = self.act(self.norm(x_edge))

        return x

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list=None):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                layers.append(nn.ReLU())
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@register('EESCit_res')     #Edge Ehanced Spatial-Channel implicit transformer
class EESC_ITres(nn.Module):
    def __init__(self,
                 backbone_spec,
                 head=8,
                 qembed_dim=[256, 128, 64, 32],
                 kembed_dim=[256, 128, 128, 256],
                 vembed_dim=[256, 128, 128, 256]
                 ):
        super().__init__()

        self.backbone = models.make(backbone_spec)
        self.dim = self.backbone.out_dim
        self.head = head

        self.EdgeEnhanceq_before = EchoEdgeModule(self.dim, self.dim)
        self.EdgeEnhanceq_after = EchoEdgeModule(self.dim, self.dim)
        self.SCEnhance_kv = nn.Sequential(
            SpatialOperation(self.dim),
            ChannelOperation(self.dim),
        )
        self.SCEnhance_k = nn.Sequential(
            SpatialOperation(self.dim),
            ChannelOperation(self.dim),
        )

        self.SCEnhance_v = nn.Sequential(
            SpatialOperation(self.dim),
            ChannelOperation(self.dim),
        )


        self.q_embed = MLP(in_dim=self.dim, out_dim=3, hidden_list=qembed_dim)
        self.k_embed = MLP(in_dim=self.dim + 2, out_dim=self.dim, hidden_list=kembed_dim)
        self.v_embed = MLP(in_dim=self.dim + 2, out_dim=self.dim, hidden_list=vembed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def get_coordinate_map(self, lr, hr_size):
        b, c, h_old, w_old = lr.shape
        h, w = hr_size
        coord_l = make_coord((h_old, w_old), flatten=False).unsqueeze(0).repeat(b,1,1,1).to(lr.device)  # b,h,w,2
        coord_h = make_coord((h, w), flatten=False).unsqueeze(0).repeat(b,1,1,1).to(lr.device)  # b,sh,sw,2

        rel_coord = coord_h - F.interpolate(coord_l.permute(0,3,1,2), size=hr_size, mode='nearest').permute(0,2,3,1)
        rel_coord[:, 0, :, :] *= h
        rel_coord[:, 1, :, :] *= w
        rel_coord = rel_coord.to(lr.device)

        cell = torch.ones(2)
        cell[0] *= 2. / h
        cell[1] *= 2. / w
        cell = cell.unsqueeze(0).to(lr.device)
        cell = cell.repeat(b, h*w, 1)

        return rel_coord.contiguous().detach(), coord_l.contiguous().detach(), coord_h.contiguous().detach(), cell.contiguous().detach()

    def forward(self, x, out_size):
        res = x
        b,c,h,w = x.shape
        sh, sw = out_size
        q = sh * sw
        rel_coord, coord_l, coord_h, cell = self.get_coordinate_map(x, out_size)

        z = self.backbone(x)

        feat_q = self.EdgeEnhanceq_before(z)
        feat_kv = self.SCEnhance_kv(z)

        # query   b,c,h,w -> b,c,sh,sw -> b,sh,sw,c -> b,q,c//h,h
        feat_q = F.grid_sample(
            feat_q,
            coord_h.flip(-1),
            mode='bilinear',
            align_corners=False
        )
        feat_q = self.EdgeEnhanceq_after(feat_q).permute(0, 2, 3, 1)
        feat_q = feat_q.reshape(b, q, self.dim // self.head, self.head)

        # key   b,c,h,w -> b,c,sh,sw -> b,sh,sw,c -> b,q,c -> b,q,h,c//h
        # wk b,q,c+2 -> b,q,c
        feat_k = F.grid_sample(
            feat_kv,
            coord_h.flip(-1),
            mode='nearest',
            align_corners=False
        )
        feat_k = self.SCEnhance_k(feat_k).permute(0, 2, 3, 1)
        feat_k = feat_k.reshape(b, q, -1)
        weight_k = torch.cat([feat_k, rel_coord.view(b,-1,2)], dim=-1)
        weight_k = self.k_embed(weight_k.reshape(b * q, -1)).reshape(b, q, -1)
        feat_k = (feat_k * weight_k).reshape(b, q, self.head, self.dim // self.head)

        # b,q,c//h, c//h
        attn = self.softmax((feat_q @ feat_k) / np.sqrt(self.dim // self.head))

        # value   b,c,h,w -> b,c,sh,sw -> b,sh,sw,c -> b,q,c -> b,q,h,c//h
        # wv b,q,c+2 -> b,q,c
        feat_v = F.grid_sample(
            feat_kv,
            coord_h.flip(-1),
            mode='nearest',
            align_corners=False
        )
        feat_v = self.SCEnhance_v(feat_v).permute(0, 2, 3, 1)
        feat_v = feat_v.reshape(b, q, -1)
        weight_v = torch.cat([feat_v, cell], dim=-1)
        weight_v = self.v_embed(weight_v.reshape(b * q, -1)).reshape(b, q, -1)
        feat_v = (feat_v * weight_v).reshape(b, q, self.dim // self.head, self.head)

        # b,q,c//h, h -> b,q,3
        attn = attn @ feat_v
        attn = self.q_embed(attn.reshape(b * q, -1)).reshape(b, q, -1)

        pred = attn.contiguous().reshape(b, sh, sw, -1).permute(0, 3, 1, 2)
        res = F.grid_sample(res, coord_h.flip(-1), mode='bilinear', \
                            padding_mode='border', align_corners=False)

        return pred+res


