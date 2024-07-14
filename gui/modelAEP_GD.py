import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


# 16 -> 32
class discriminator_l0(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(discriminator_l0, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,              self.d_dim * 1, 3, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim * 1, self.d_dim * 2, 3, stride=1, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim * 2, self.d_dim * 4, 3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim * 4, self.z_dim,     1, stride=1, padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = torch.sigmoid(out)

        return out


# 32 -> 64
class discriminator_l1(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(discriminator_l1, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,              self.d_dim * 1, 3, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim * 1, self.d_dim * 2, 3, stride=1, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim * 2, self.d_dim * 4, 3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim * 4, self.d_dim * 8, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim * 8, self.z_dim,     1, stride=1, padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = torch.sigmoid(out)

        return out


# 64 -> 128
class discriminator_l2(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(discriminator_l2, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,              self.d_dim * 1, 4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim * 1, self.d_dim * 2, 3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim * 2, self.d_dim * 4, 3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim * 4, self.d_dim * 8, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim * 8, self.d_dim * 8, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim * 8, self.z_dim,     1, stride=1, padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = torch.sigmoid(out)

        return out


# 128 -> 256
class discriminator_l3(nn.Module):
    def __init__(self, d_dim, z_dim):
        super(discriminator_l3, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(1,              self.d_dim * 1,  4, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.d_dim * 1, self.d_dim * 2,  3, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim * 2, self.d_dim * 4,  3, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim * 4, self.d_dim * 8,  3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim * 8, self.d_dim * 16, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim * 16, self.z_dim,     1, stride=1, padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = torch.sigmoid(out)

        return out


# 16 -> 32 -> 64 -> 128
class pyramid_generator_shaddr_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim, p_dim):
        super(pyramid_generator_shaddr_x8, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.p_dim = p_dim

        style_codes = torch.zeros((self.z_dim, self.prob_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        # backbone
        self.conv_00 = nn.Conv3d(self.p_dim + 1 + self.z_dim,              self.g_dim * 1,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_01 = nn.Conv3d(self.g_dim * 1 + self.z_dim,              self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_02 = nn.Conv3d(self.g_dim * 2 + self.z_dim,              self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_03 = nn.Conv3d(self.g_dim * 4 + self.z_dim,              self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_04 = nn.Conv3d(self.g_dim * 8 + self.z_dim,              self.g_dim * 16, 3, stride=1, dilation=1, padding=1, bias=True)

        # 16 -> 32
        self.conv_10 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 16 + self.z_dim,    self.g_dim * 8,  4, stride=2, padding=1, bias=True)
        self.conv_11 = nn.Conv3d(self.g_dim * 8,                                        self.g_dim * 8,  3, stride=1, padding=1, bias=True)
        self.conv_o1 = nn.Conv3d(self.g_dim * 8,                                        1,               3, stride=1, padding=1, bias=True)

        # 32 -> 64
        self.conv_20 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 8 + self.z_dim,     self.g_dim * 4,  4, stride=2, padding=1, bias=True)
        self.conv_21 = nn.Conv3d(self.g_dim * 4,                                        self.g_dim * 4,  3, stride=1, padding=1, bias=True)
        self.conv_o2 = nn.Conv3d(self.g_dim * 4,                                        1,               3, stride=1, padding=1, bias=True)

        # 64 -> 128
        self.conv_30 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 4 + self.z_dim,     self.g_dim * 2,  4, stride=2, padding=1, bias=True)
        self.conv_31 = nn.Conv3d(self.g_dim * 2,                                        self.g_dim * 2,  3, stride=1, padding=1, bias=True)
        self.conv_o3 = nn.Conv3d(self.g_dim * 2,                                        1,               3, stride=1, padding=1, bias=True)

    def forward(self, geo_voxels, seg_voxels, z, is_training=False):

        out = torch.cat([geo_voxels, seg_voxels], dim=1)

        # -------------------- backbone --------------------
        zs = z

        out = torch.cat([out, zs], dim=1)
        out = self.conv_00(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_01(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_02(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_03(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_04(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        # -------------------- 16 -> 32 --------------------
        seg_vox = seg_voxels
        out = torch.cat([out, seg_vox], dim=1)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_11(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l1 = self.conv_o1(out)
        out_l1 = torch.max(torch.min(out_l1, out_l1 * 0.002 + 0.998), out_l1 * 0.002)

        # -------------------- 32 -> 64 --------------------
        seg_vox = F.interpolate(seg_voxels, scale_factor=2, mode='nearest')
        out = torch.cat([out, seg_vox], dim=1)
        zs = F.interpolate(z, scale_factor=2, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_20(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_21(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l2 = self.conv_o2(out)
        out_l2 = torch.max(torch.min(out_l2, out_l2 * 0.002 + 0.998), out_l2 * 0.002)

        # -------------------- 64 -> 128 --------------------
        seg_vox = F.interpolate(seg_voxels, scale_factor=4, mode='nearest')
        out = torch.cat([out, seg_vox], dim=1)
        zs = F.interpolate(z, scale_factor=4, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_30(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_31(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l3 = self.conv_o3(out)
        out_l3 = torch.max(torch.min(out_l3, out_l3 * 0.002 + 0.998), out_l3 * 0.002)

        return out_l1, out_l2, out_l3


# 16 -> 32 -> 64 -> 128
class pyramid_generator_shaddr_x16(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim, p_dim):
        super(pyramid_generator_shaddr_x16, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.p_dim = p_dim

        style_codes = torch.zeros((self.z_dim, self.prob_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        # backbone
        self.conv_00 = nn.Conv3d(self.p_dim + 1 + self.z_dim,              self.g_dim * 1,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_01 = nn.Conv3d(self.g_dim * 1 + self.z_dim,              self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_02 = nn.Conv3d(self.g_dim * 2 + self.z_dim,              self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_03 = nn.Conv3d(self.g_dim * 4 + self.z_dim,              self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_04 = nn.Conv3d(self.g_dim * 8 + self.z_dim,              self.g_dim * 16, 3, stride=1, dilation=1, padding=1, bias=True)

        # 16 -> 32
        self.conv_10 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 16 + self.z_dim,    self.g_dim * 8,  4, stride=2, padding=1, bias=True)
        self.conv_11 = nn.Conv3d(self.g_dim * 8,                                        self.g_dim * 8,  3, stride=1, padding=1, bias=True)
        self.conv_o1 = nn.Conv3d(self.g_dim * 8,                                        1,               3, stride=1, padding=1, bias=True)

        # 32 -> 64
        self.conv_20 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 8 + self.z_dim,     self.g_dim * 4,  4, stride=2, padding=1, bias=True)
        self.conv_21 = nn.Conv3d(self.g_dim * 4,                                        self.g_dim * 4,  3, stride=1, padding=1, bias=True)
        self.conv_o2 = nn.Conv3d(self.g_dim * 4,                                        1,               3, stride=1, padding=1, bias=True)

        # 64 -> 128
        self.conv_30 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 4 + self.z_dim,     self.g_dim * 2,  4, stride=2, padding=1, bias=True)
        self.conv_31 = nn.Conv3d(self.g_dim * 2,                                        self.g_dim * 2,  3, stride=1, padding=1, bias=True)
        self.conv_o3 = nn.Conv3d(self.g_dim * 2,                                        1,               3, stride=1, padding=1, bias=True)

        # 128 -> 256
        self.conv_40 = nn.ConvTranspose3d(self.p_dim + self.g_dim * 2 + self.z_dim,     self.g_dim * 1,  4, stride=2, padding=1, bias=True)
        self.conv_o4 = nn.Conv3d(self.g_dim * 1,                                        1,               3, stride=1, padding=1, bias=True)

    def forward(self, geo_voxels, seg_voxels, z, is_training=False):

        out = torch.cat([geo_voxels, seg_voxels], dim=1)

        # -------------------- backbone --------------------
        zs = z

        out = torch.cat([out, zs], dim=1)
        out = self.conv_00(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_01(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_02(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_03(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_04(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        # -------------------- 16 -> 32 --------------------
        seg_vox = seg_voxels
        out = torch.cat([out, seg_vox], dim=1)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_11(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l1 = self.conv_o1(out)
        out_l1 = torch.max(torch.min(out_l1, out_l1 * 0.002 + 0.998), out_l1 * 0.002)

        # -------------------- 32 -> 64 --------------------
        seg_vox = F.interpolate(seg_voxels, scale_factor=2, mode='nearest')
        out = torch.cat([out, seg_vox], dim=1)
        zs = F.interpolate(z, scale_factor=2, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_20(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_21(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l2 = self.conv_o2(out)
        out_l2 = torch.max(torch.min(out_l2, out_l2 * 0.002 + 0.998), out_l2 * 0.002)

        # -------------------- 64 -> 128 --------------------
        seg_vox = F.interpolate(seg_voxels, scale_factor=4, mode='nearest')
        out = torch.cat([out, seg_vox], dim=1)
        zs = F.interpolate(z, scale_factor=4, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_30(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_31(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l3 = self.conv_o3(out)
        out_l3 = torch.max(torch.min(out_l3, out_l3 * 0.002 + 0.998), out_l3 * 0.002)

        # ------------------- 128 -> 256 --------------------
        seg_vox = F.interpolate(seg_voxels, scale_factor=8, mode='nearest')
        out = torch.cat([out, seg_vox], dim=1)
        zs = F.interpolate(z, scale_factor=8, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_40(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l4 = self.conv_o4(out)
        out_l4 = torch.max(torch.min(out_l4, out_l4 * 0.002 + 0.998), out_l4 * 0.002)

        return out_l1, out_l2, out_l3, out_l4


# 16 -> 32 -> 64 -> 128
class pyramid_generator_shaddr_x8_wo_parts(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(pyramid_generator_shaddr_x8_wo_parts, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        style_codes = torch.zeros((self.z_dim, self.prob_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        # backbone
        self.conv_00 = nn.Conv3d(1 + self.z_dim,                           self.g_dim * 1,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_01 = nn.Conv3d(self.g_dim * 1 + self.z_dim,              self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_02 = nn.Conv3d(self.g_dim * 2 + self.z_dim,              self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_03 = nn.Conv3d(self.g_dim * 4 + self.z_dim,              self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_04 = nn.Conv3d(self.g_dim * 8 + self.z_dim,              self.g_dim * 16, 3, stride=1, dilation=1, padding=1, bias=True)

        # 16 -> 32
        self.conv_10 = nn.ConvTranspose3d(self.g_dim * 16 + self.z_dim,    self.g_dim * 8,  4, stride=2, padding=1, bias=True)
        self.conv_11 = nn.Conv3d(self.g_dim * 8,                           self.g_dim * 8,  3, stride=1, padding=1, bias=True)
        self.conv_o1 = nn.Conv3d(self.g_dim * 8,                           1,               3, stride=1, padding=1, bias=True)

        # 32 -> 64
        self.conv_20 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,     self.g_dim * 4,  4, stride=2, padding=1, bias=True)
        self.conv_21 = nn.Conv3d(self.g_dim * 4,                           self.g_dim * 4,  3, stride=1, padding=1, bias=True)
        self.conv_o2 = nn.Conv3d(self.g_dim * 4,                           1,               3, stride=1, padding=1, bias=True)

        # 64 -> 128
        self.conv_30 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,     self.g_dim * 2,  4, stride=2, padding=1, bias=True)
        self.conv_31 = nn.Conv3d(self.g_dim * 2,                           self.g_dim * 2,  3, stride=1, padding=1, bias=True)
        self.conv_o3 = nn.Conv3d(self.g_dim * 2,                           1,               3, stride=1, padding=1, bias=True)

    def forward(self, geo_voxels, z, is_training=False):

        out = geo_voxels

        # -------------------- backbone --------------------
        zs = z

        out = torch.cat([out, zs], dim=1)
        out = self.conv_00(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_01(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_02(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_03(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out, zs], dim=1)
        out = self.conv_04(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        # -------------------- 16 -> 32 --------------------

        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_11(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l1 = self.conv_o1(out)
        out_l1 = torch.max(torch.min(out_l1, out_l1 * 0.002 + 0.998), out_l1 * 0.002)

        # -------------------- 32 -> 64 --------------------

        zs = F.interpolate(z, scale_factor=2, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_20(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_21(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l2 = self.conv_o2(out)
        out_l2 = torch.max(torch.min(out_l2, out_l2 * 0.002 + 0.998), out_l2 * 0.002)

        # -------------------- 64 -> 128 --------------------

        zs = F.interpolate(z, scale_factor=4, mode='nearest')

        out = torch.cat([out, zs], dim=1)
        out = self.conv_30(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_31(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_l3 = self.conv_o3(out)
        out_l3 = torch.max(torch.min(out_l3, out_l3 * 0.002 + 0.998), out_l3 * 0.002)

        return out_l1, out_l2, out_l3


class generator_halfsize_x8_decorgan(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim, p_dim):
        super(generator_halfsize_x8_decorgan, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim
        self.p_dim = p_dim

        style_codes = torch.zeros((self.z_dim, self.prob_dim))
        self.style_codes = nn.Parameter(style_codes)
        nn.init.constant_(self.style_codes, 0.0)

        self.conv_0 = nn.Conv3d(self.p_dim+1+self.z_dim,  self.g_dim,    3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim+self.z_dim,    self.g_dim*2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim*2+self.z_dim,  self.g_dim*4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim*4+self.z_dim,  self.g_dim*8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim*8+self.z_dim,  self.g_dim*16,  3, stride=1, dilation=1, padding=1, bias=True)

        self.conv_5 = nn.ConvTranspose3d(self.g_dim*16,  self.g_dim*8, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim*8,  self.g_dim*4,  3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim*4,  self.g_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim*2,  self.g_dim*2,  3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim*2,  self.g_dim,   4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim,   1,             3, stride=1, padding=1, bias=True)

    def forward(self, geo_voxels, seg_voxels, z, mask_, is_training=False):

        out = torch.cat([geo_voxels, seg_voxels], dim=1)
        mask = F.interpolate(mask_, scale_factor=8, mode='nearest')

        zs = z

        out = torch.cat([out,zs], dim=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out,zs], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out,zs], dim=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out,zs], dim=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([out,zs], dim=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)

        out = out*mask

        return out


if __name__ == '__main__':
    model = discriminator_l2(d_dim=32, z_dim=16)
    # geo_voxels = torch.rand(1, 1, 16, 16, 16).cuda()
    # seg_voxels = torch.rand(1, 5, 16, 16, 16).cuda()
    # z = torch.rand(1, 8, 16, 16, 16).cuda()
    # model = pyramid_generator_shaddr_x8(g_dim=32, prob_dim=16, z_dim=8, p_dim=5).cuda()
    # with torch.no_grad():
    #     out = model(geo_voxels, seg_voxels, z)
    # print(out[-1].shape)
    print("Num param: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
