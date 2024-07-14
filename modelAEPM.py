import os
import time
import math
import random
import numpy as np
import cv2
import h5py
import mcubes
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils import *
from modelAEP_GD import *


SETTINGS = {
    "03001627": {
        "combination": False,
        "rotation": False,
        "num_parts": 5,  # 1: back, 2: seat, 3: leg, 4: armrest, 5: stretcher
    },
    "04379243": {
        "combination": False,
        "rotation": False,
        "num_parts": 2,  # 2: seat, 3: leg, same as chair
    },
    "03593526": {
        "combination": True,
        "rotation": True,
        "num_parts": 2,  # 1: pot, 2: leaves
    },
    "00000000": {
        "combination": True,
        "rotation": True,
        "num_parts": 2,  # 1: body, 2: roof
    },
    "00000001": {
        "combination": True,
        "rotation": True,
        "num_parts": 2,  # 1: bottom part, 2: top part
    },
}


class IM_AE(object):
    def __init__(self, config):
        self.real_size = 256
        self.mask_margin = 16

        self.g_dim = 32
        self.d_dim = 32
        self.z_dim = 8
        self.param_alpha = config.alpha
        self.param_beta = config.beta

        self.input_size = config.input_size
        self.output_size = config.output_size

        if self.input_size == 16 and self.output_size == 128:
            self.upsample_rate = 8
            self.upsample_level = 3
        elif self.input_size == 16 and self.output_size == 256:
            self.upsample_rate = 16
            self.upsample_level = 4
        else:
            print("ERROR: invalid input/output size!")
            exit(-1)

        self.category_id = config.data_dir.split('/')[-2].split('_')
        self.num_parts = max([SETTINGS[category_id]["num_parts"] for category_id in self.category_id])
        self.num_augmentation = 2560

        self.sampling_threshold = 0.4

        self.render_view_id = 0
        # self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_model = config.checkpoint_model
        self.data_dir = config.data_dir

        self.data_style = config.data_style
        self.data_content = config.data_content

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # load data
        print("preprocessing - start")

        if os.path.exists("splits/" + self.data_style + ".txt"):

            # load data
            fin = open("splits/" + self.data_style + ".txt")
            self.styleset_names = [name.strip() for name in fin.readlines()]
            fin.close()
            self.styleset_len = len(self.styleset_names)
            self.voxel_style = []
            self.Dmask_style = []
            self.input_style = []
            self.Smask_style = []
            self.voxel_raw = []
            self.segmt_raw = []
            self.has_parts_style = [[] for _ in range(self.num_parts)]

            self.input_content = []
            self.Smask_content = []
            self.Dmask_content = []
            self.Jmask_content = []

            if config.train:
                for i in range(self.styleset_len):
                    vox_path = os.path.join(self.data_dir, self.styleset_names[i] + "/model_depth_fusion.binvox")
                    seg_path = os.path.join(self.data_dir, self.styleset_names[i] + "/segmentation_voxel.hdf5")
                    data_dict = h5py.File(seg_path, 'r')
                    seg_raw = data_dict["segmentation"][:]  # note that seg_vox is already (256, 256, 256)
                    data_dict.close()

                    if self.output_size == 128:
                        geo_raw = get_vox_from_binvox_1over2_return_small(vox_path).astype(np.uint8)
                        seg_raw = self.get_downsampled_segmentation_voxel(seg_raw, downsample_levels=1)
                    elif self.output_size == 256:
                        geo_raw = get_vox_from_binvox(vox_path).astype(np.uint8)
                    else:
                        print("ERROR: invalid output size!")
                        exit(-1)

                    self.segmt_raw.append(seg_raw)
                    self.voxel_raw.append(geo_raw)

                    for part_id in range(self.num_parts):
                        if part_id + 1 in np.unique(seg_raw):
                            self.has_parts_style[part_id].append(i)

                    xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(geo_raw)
                    tmp_geo = self.crop_voxel(geo_raw, xmin, xmax, ymin, ymax, zmin, zmax)
                    tmp_seg = self.crop_voxel(seg_raw, xmin, xmax, ymin, ymax, zmin, zmax)

                    tmp_input, tmp_Smask, tmp_Dmask, tmp_Jmask = self.get_voxel_input_Smask_Dmasks_Jmasks(tmp_geo, tmp_seg)
                    self.input_style.append(tmp_input)
                    self.Smask_style.append(tmp_Smask)
                    self.Dmask_style.append(tmp_Dmask)
                    self.voxel_style.append(self.get_style_voxel_lod(tmp_geo))

                    self.input_content.append(tmp_input)
                    self.Smask_content.append(tmp_Smask)
                    self.Dmask_content.append(tmp_Dmask)
                    self.Jmask_content.append(tmp_Jmask)

                    print("preprocessing style - " + str(i + 1) + "/" + str(self.styleset_len) + " " + self.styleset_names[i] + " - " + str(np.unique(seg_raw)))
                print(self.has_parts_style)

                # augmentation
                count = 0
                num_augmentation_per_style_shape = self.num_augmentation // self.styleset_len
                for i in range(self.styleset_len):

                    current_style_category = self.styleset_names[i].split('_')[0]
                    combination = SETTINGS[current_style_category]["combination"]
                    rotation = SETTINGS[current_style_category]["rotation"]

                    for j in range(num_augmentation_per_style_shape):
                        augmented_geo, augmented_seg = self.get_voxel_augmentation(i, combination, rotation)
                        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(augmented_geo)
                        tmp_geo = self.crop_voxel(augmented_geo, xmin, xmax, ymin, ymax, zmin, zmax)
                        tmp_seg = self.crop_voxel(augmented_seg, xmin, xmax, ymin, ymax, zmin, zmax)

                        tmp_input, tmp_Smask, tmp_Dmask, tmp_Jmask = self.get_voxel_input_Smask_Dmasks_Jmasks(tmp_geo, tmp_seg)
                        self.input_content.append(tmp_input)
                        self.Smask_content.append(tmp_Smask)
                        self.Dmask_content.append(tmp_Dmask)
                        self.Jmask_content.append(tmp_Jmask)
                        count = count + 1
                        print("augmenting style " + str(i + 1) + " - " + str(count) + "/" + str(self.num_augmentation) + " - " + str(tmp_input.shape))

                self.dataset_len = len(self.input_content)

        else:
            print("ERROR: cannot load styleset txt: " + "splits/" + self.data_style + ".txt")
            exit(-1)

        # build model
        self.discriminator = []
        self.discriminator.append(discriminator_l0(self.d_dim, self.styleset_len + 1).to(self.device))
        self.discriminator.append(discriminator_l1(self.d_dim, self.styleset_len + 1).to(self.device))
        self.discriminator.append(discriminator_l3(self.d_dim, self.styleset_len + 1).to(self.device))
        discriminator_params = (list(self.discriminator[0].parameters()) + list(self.discriminator[1].parameters()) +
                                list(self.discriminator[2].parameters()))
        if self.upsample_level == 4:
            self.discriminator.append(discriminator_l3(self.d_dim, self.styleset_len + 1).to(self.device))
            discriminator_params = (list(self.discriminator[0].parameters()) + list(self.discriminator[1].parameters()) +
                                    list(self.discriminator[2].parameters()) + list(self.discriminator[3].parameters()))

        if self.input_size == 16 and self.output_size == 128:
            self.generator = pyramid_generator_shaddr_x8(self.g_dim, self.styleset_len, self.z_dim, self.num_parts)
            # self.generator = pyramid_generator_shaddr_x8_wo_parts(self.g_dim, self.styleset_len, self.z_dim)
        elif self.input_size == 16 and self.output_size == 256:
            self.generator = pyramid_generator_shaddr_x16(self.g_dim, self.styleset_len, self.z_dim, self.num_parts)
            # self.generator = pyramid_generator_shaddr_x16_wo_parts(self.g_dim, self.styleset_len, self.z_dim)
        self.generator.to(self.device)

        self.optimizer_d = torch.optim.Adam(discriminator_params, lr=0.0001)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001)

        # checkpoint_dir   - checkpoint root dir
        # checkpoint_path  - path for saving checkpoint, e.g. checkpoint_dir/{self.data_style}_ae
        # checkpoint_model - path for loading checkpoint, e.g. {self.data_style}_ae/IM_AE.model-continuous-19.pth
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "{}_ae".format(self.data_style))
        self.print_param()

    def get_voxel_input_Smask_Dmasks_Jmasks(self, geometry_vox, segmentation_vox):

        geo_tensor = torch.from_numpy(geometry_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        seg_tensor = torch.from_numpy(segmentation_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()

        # --------------------------------- input geometry ---------------------------------
        smallmaskx_tensor = F.max_pool3d(geo_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        input_vox = np.round(smallmaskx_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)

        # --------------------------------- Smask ---------------------------------
        small_seg = self.get_downsampled_segmentation_voxel(segmentation_vox, downsample_levels=self.upsample_level)

        # Nearest neighbor
        labelled_points = np.stack((np.where(input_vox == 1))).T
        labels = small_seg[labelled_points[:, 0], labelled_points[:, 1], labelled_points[:, 2]]
        unlabelled_points = np.stack((np.where(input_vox == 0))).T
        labelled_tree = KDTree(labelled_points)
        _, inds = labelled_tree.query(unlabelled_points, k=1)
        small_seg[unlabelled_points[:, 0], unlabelled_points[:, 1], unlabelled_points[:, 2]] = labels[inds[:, 0]]
        assert 0 not in np.unique(small_seg)

        smallmasks_tensor = torch.from_numpy(small_seg).to(self.device).unsqueeze(0).unsqueeze(0).float()
        _, _, dimx, dimy, dimz = smallmasks_tensor.size()
        smallmasks_tensor = F.one_hot(smallmasks_tensor.contiguous().view(dimx * dimy * dimz).long(),
                                      num_classes=self.num_parts + 1)[:, 1:].contiguous().permute(1, 0).view(1, self.num_parts, dimx, dimy, dimz).float()
        input_seg = np.round(smallmasks_tensor.detach().cpu().numpy()[0]).astype(np.uint8)

        # --------------------------------- Dmask ---------------------------------
        Dmasks = []
        smallmasks_tensor = smallmaskx_tensor * smallmasks_tensor

        # Dmasks for 32 from 16, 16 -> 32 -> 32-3-3=26
        crop_margin = 3
        dmask_tensor = F.interpolate(smallmasks_tensor, scale_factor=2, mode='nearest')
        dmask_tensor = dmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        dmask = np.round(dmask_tensor.detach().cpu().numpy()[0]).astype(np.uint8)
        Dmasks.append(dmask)

        # Dmasks for 64 from 16, 16 -> 64 -> 64-4-4=56
        crop_margin = 4
        dmask_tensor = F.interpolate(smallmasks_tensor, scale_factor=4, mode='nearest')
        dmask_tensor = dmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        dmask = np.round(dmask_tensor.detach().cpu().numpy()[0]).astype(np.uint8)
        Dmasks.append(dmask)

        # Dmask for 128 from 16, 16 -> 64 -> 64-4-4=56
        crop_margin = 4
        dmask_tensor = F.interpolate(smallmasks_tensor, scale_factor=4, mode='nearest')
        dmask_tensor = dmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        dmask = np.round(dmask_tensor.detach().cpu().numpy()[0]).astype(np.uint8)
        Dmasks.append(dmask)

        # Dmask for 256 from 16, 16 -> 128 -> 128-4-4=120
        # crop_margin = 4
        # dmask_tensor = F.interpolate(smallmasks_tensor, scale_factor=8, mode='nearest')
        # dmask_tensor = dmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        # dmask = np.round(dmask_tensor.detach().cpu().numpy()[0]).astype(np.uint8)
        # Dmasks.append(dmask)

        # --------------------------------- Jmask ---------------------------------
        Jmasks = []
        _, _, dimx, dimy, dimz = seg_tensor.shape
        seg_tensor_one_hot = F.one_hot(seg_tensor.contiguous().view(dimx * dimy * dimz).long(),
                                       num_classes=self.num_parts + 1)[:, 1:].contiguous().permute(1, 0).view(1, self.num_parts, dimx, dimy, dimz).float()
        dilated_tensor = F.max_pool3d(seg_tensor_one_hot * geo_tensor, kernel_size=5, stride=1, padding=2)
        joints_tensor = (torch.sum(dilated_tensor, dim=1, keepdim=True) > 1).float()
        smalljoints_tensor = F.max_pool3d(joints_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)

        # Jmasks for 32 from 16, 16 -> 32 -> 32-3-3=26
        crop_margin = 3
        jmask_tensor = F.interpolate(smalljoints_tensor, scale_factor=2, mode='nearest')
        jmask_tensor = jmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        jmask = np.round(jmask_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        Jmasks.append(jmask)

        # Jmasks for 64 from 16, 16 -> 64 -> 64-4-4=56
        crop_margin = 4
        jmask_tensor = F.interpolate(smalljoints_tensor, scale_factor=4, mode='nearest')
        jmask_tensor = jmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        jmask = np.round(jmask_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        Jmasks.append(jmask)

        # Jmasks for 128 from 16, 16 -> 64 -> 64-4-4=56
        crop_margin = 4
        jmask_tensor = F.interpolate(smalljoints_tensor, scale_factor=4, mode='nearest')
        jmask_tensor = jmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        jmask = np.round(jmask_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        Jmasks.append(jmask)

        # Jmasks for 256 from 16, 16 -> 128 -> 128-4-4=120
        # crop_margin = 4
        # jmask_tensor = F.interpolate(smalljoints_tensor, scale_factor=8, mode='nearest')
        # jmask_tensor = jmask_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        # jmask = np.round(jmask_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        # Jmasks.append(jmask)

        return input_vox, input_seg, Dmasks, Jmasks

    def get_style_voxel_lod(self, vox):

        # input style shape 128
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()

        style_voxels = []

        # 32
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=4, stride=4, padding=0)
        smallmaskx = np.round(smallmaskx_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        style_voxels.append(self.progressive_gaussian_blur(smallmaskx.astype(np.float32), iterations=4))

        # 64
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=2, stride=2, padding=0)
        smallmaskx = np.round(smallmaskx_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        style_voxels.append(self.progressive_gaussian_blur(smallmaskx.astype(np.float32), iterations=4))

        # 128
        # smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=2, stride=2, padding=0)
        # smallmaskx = np.round(smallmaskx_tensor.detach().cpu().numpy()[0, 0]).astype(np.uint8)
        # style_voxels.append(self.progressive_gaussian_blur(smallmaskx.astype(np.float32), iterations=4))

        # 256
        style_voxels.append(self.progressive_gaussian_blur(vox.astype(np.float32), iterations=4))

        return style_voxels

    def progressive_gaussian_blur(self, vox, iterations=1):

        gaussian_vox = gaussian_filter(vox.astype(np.float32), sigma=1.0)

        for _ in range(iterations):
            binarized_gaussian_vox = (gaussian_vox > self.sampling_threshold).astype(np.uint8)
            diff = vox - binarized_gaussian_vox
            diff_gaussian = gaussian_filter(diff.astype(np.float32), sigma=1.0)
            gaussian_vox = gaussian_vox + diff_gaussian

        gaussian_vox = np.clip(gaussian_vox, 0.0, 1.0)

        return gaussian_vox

    def get_downsampled_segmentation_voxel(self, segmentation_vox, downsample_levels=0):

        # the vox_seg should be one-hot encoding
        dimx, dimy, dimz = segmentation_vox.shape
        voxel_low = F.one_hot(torch.from_numpy(segmentation_vox).contiguous().view(-1).long(),
                              num_classes=self.num_parts + 1)[:, 1:].view(dimx, dimy, dimz, self.num_parts).float().numpy()

        for l in range(downsample_levels):
            voxel_high = voxel_low
            voxel_low = np.zeros([dimx // 2 ** (l + 1), dimy // 2 ** (l + 1), dimz // 2 ** (l + 1), self.num_parts], np.uint8)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        voxel_low = voxel_low + voxel_high[i::2, j::2, k::2]

        voxelM = (np.max(voxel_low, -1, keepdims=True) > 0).astype(np.uint8)
        voxelI = np.argmax(voxel_low, -1)
        voxelO = np.zeros([dimx // 2 ** downsample_levels, dimy // 2 ** downsample_levels, dimz // 2 ** downsample_levels, self.num_parts], np.uint8)
        for i in range(self.num_parts):
            voxelO[..., i] = (voxelI == i)
        voxelO = voxelO * voxelM  # target one-hot encoding
        voxelO_label = np.argmax(voxelO, axis=-1) + 1  # target label encoding

        return voxelO_label

    def _get_voxel_augmentation(self, geometry_vox, segmentation_vox, rotation=False):

        max_size = self.output_size * 0.7

        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(geometry_vox)
        geometry_vox = self.crop_voxel(geometry_vox, xmin, xmax, ymin, ymax, zmin, zmax, False)
        segmentation_vox = self.crop_voxel(segmentation_vox, xmin, xmax, ymin, ymax, zmin, zmax, False)

        geometry_tensor = torch.from_numpy(geometry_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        _, _, dimx, dimy, dimz = geometry_tensor.size()
        scale_dimx, scale_dimy, scale_dimz = np.random.uniform(low=0.7, high=2.0, size=3)
        new_dimx = int(np.round(dimx * scale_dimx))
        new_dimy = int(np.round(dimy * scale_dimy))
        new_dimz = int(np.round(dimz * scale_dimz))
        max_dim_size = max(new_dimx, new_dimy, new_dimz)
        if max_dim_size > max_size:
            new_dimx = int(np.round(new_dimx * max_size / max_dim_size))
            new_dimy = int(np.round(new_dimy * max_size / max_dim_size))
            new_dimz = int(np.round(new_dimz * max_size / max_dim_size))

        augmented_geometry_tensor = F.interpolate(geometry_tensor, size=(new_dimx, new_dimy, new_dimz), mode='trilinear')
        augmented_geometry_tensor = (augmented_geometry_tensor > 0.3).float()
        segmentation_tensor = torch.from_numpy(segmentation_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        augmented_segmentation_tensor = F.interpolate(segmentation_tensor, size=(new_dimx, new_dimy, new_dimz), mode='trilinear')

        augmented_geometry = augmented_geometry_tensor.detach().cpu().numpy()[0, 0]
        augmented_segmentation = augmented_segmentation_tensor.detach().cpu().numpy()[0, 0]

        padx_left = (self.output_size - new_dimx) // 2
        pady_left = (self.output_size - new_dimy) // 2
        padz_left = (self.output_size - new_dimz) // 2
        padx_right = self.output_size - new_dimx - padx_left
        pady_right = self.output_size - new_dimy - pady_left
        padz_right = self.output_size - new_dimz - padz_left
        augmented_geometry = np.pad(augmented_geometry, ((padx_left, padx_right), (pady_left, pady_right), (padz_left, padz_right)), 'constant', constant_values=0)
        augmented_segmentation = np.pad(augmented_segmentation, ((padx_left, padx_right), (pady_left, pady_right), (padz_left, padz_right)), 'constant', constant_values=0)

        augmented_geometry = np.round(augmented_geometry).astype(np.uint8)
        augmented_segmentation = np.round(augmented_segmentation).astype(np.uint8)

        if rotation:
            rotation_degree = np.random.randint(4)
            augmented_geometry = np.rot90(augmented_geometry, k=rotation_degree, axes=(0, 2)).copy()
            augmented_segmentation = np.rot90(augmented_segmentation, k=rotation_degree, axes=(0, 2)).copy()

        return augmented_geometry, augmented_segmentation

    def get_voxel_augmentation(self, idx, combination, rotation):

        if not combination:
            geo_vox = self.voxel_raw[idx]
            seg_vox = self.segmt_raw[idx]
            augmented_geo, augmented_seg = self._get_voxel_augmentation(geo_vox, seg_vox, rotation)

            return augmented_geo, augmented_seg
        else:
            geo_vox = self.voxel_raw[idx]
            seg_vox = self.segmt_raw[idx]
            augmented_geo_org, augmented_seg_org = self._get_voxel_augmentation(geo_vox, seg_vox, rotation)

            if np.random.randint(2):
                return augmented_geo_org, augmented_seg_org

            augmented_geo_tmp, augmented_seg_tmp = self._get_voxel_augmentation(geo_vox, seg_vox, rotation)

            dimx1, dimy1, dimz1 = augmented_geo_org.shape
            dimx2, dimy2, dimz2 = augmented_geo_tmp.shape
            geo_tmp1 = np.zeros((dimx1 + dimx2, dimy1 + dimy2, dimz1 + dimz2))
            geo_tmp2 = np.zeros((dimx1 + dimx2, dimy1 + dimy2, dimz1 + dimz2))
            seg_tmp1 = np.zeros((dimx1 + dimx2, dimy1 + dimy2, dimz1 + dimz2))
            seg_tmp2 = np.zeros((dimx1 + dimx2, dimy1 + dimy2, dimz1 + dimz2))

            xmin1, xmax1 = np.nonzero(np.sum(augmented_geo_org, axis=(1, 2)))[0][[0, -1]]
            xmin2, xmax2 = np.nonzero(np.sum(augmented_geo_tmp, axis=(1, 2)))[0][[0, -1]]
            x_offset = random.uniform(0.3, 0.7)
            xmin1_ = min(xmin1, xmin2)
            xmin2_ = np.random.randint(xmin1_, xmin1_ + np.round((xmax1 - xmin1) * x_offset))

            ymin1, ymax1 = np.nonzero(np.sum(augmented_geo_org, axis=(0, 2)))[0][[0, -1]]
            ymin2, ymax2 = np.nonzero(np.sum(augmented_geo_tmp, axis=(0, 2)))[0][[0, -1]]
            ymin1_ = min(ymin1, ymin2)
            ymin2_ = ymin1_

            zmin1, zmax1 = np.nonzero(np.sum(augmented_geo_org, axis=(0, 1)))[0][[0, -1]]
            zmin2, zmax2 = np.nonzero(np.sum(augmented_geo_tmp, axis=(0, 1)))[0][[0, -1]]
            z_offset = random.uniform(0.3, 0.7)
            zmin1_ = min(zmin1, zmin2)
            zmin2_ = np.random.randint(zmin1_, zmin1_ + np.round((zmax1 - zmin1) * z_offset))

            geo_tmp1[xmin1_:xmin1_ + xmax1 - xmin1 + 1,
            ymin1_:ymin1_ + ymax1 - ymin1 + 1,
            zmin1_:zmin1_ + zmax1 - zmin1 + 1] = augmented_geo_org[xmin1:xmax1 + 1, ymin1:ymax1 + 1, zmin1:zmax1 + 1]
            geo_tmp2[xmin2_:xmin2_ + xmax2 - xmin2 + 1,
            ymin2_:ymin2_ + ymax2 - ymin2 + 1,
            zmin2_:zmin2_ + zmax2 - zmin2 + 1] = augmented_geo_tmp[xmin2:xmax2 + 1, ymin2:ymax2 + 1, zmin2:zmax2 + 1]
            augmented_geo_org = np.logical_or(geo_tmp1, geo_tmp2).astype(np.uint8)

            seg_tmp1[xmin1_:xmin1_ + xmax1 - xmin1 + 1,
            ymin1_:ymin1_ + ymax1 - ymin1 + 1,
            zmin1_:zmin1_ + zmax1 - zmin1 + 1] = augmented_seg_org[xmin1:xmax1 + 1, ymin1:ymax1 + 1, zmin1:zmax1 + 1]
            seg_tmp2[xmin2_:xmin2_ + xmax2 - xmin2 + 1,
            ymin2_:ymin2_ + ymax2 - ymin2 + 1,
            zmin2_:zmin2_ + zmax2 - zmin2 + 1] = augmented_seg_tmp[xmin2:xmax2 + 1, ymin2:ymax2 + 1, zmin2:zmax2 + 1]
            geo_intersection = np.logical_and(geo_tmp1, geo_tmp2).astype(np.uint8)
            seg_intersection_avg = np.floor((seg_tmp1 + seg_tmp2) / 2 * geo_intersection).astype(np.uint8)
            augmented_seg_org = seg_tmp1 * (geo_tmp1 - geo_intersection) + seg_tmp2 * (geo_tmp2 - geo_intersection) + seg_intersection_avg

            max_size = self.output_size * 0.7
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(augmented_geo_org)
            geometry_vox = self.crop_voxel(augmented_geo_org, xmin, xmax, ymin, ymax, zmin, zmax, False)
            segmentation_vox = self.crop_voxel(augmented_seg_org, xmin, xmax, ymin, ymax, zmin, zmax, False)

            dimx, dimy, dimz = geometry_vox.shape
            max_dim_size = max(dimx, dimy, dimz)
            if max_dim_size > max_size:
                dimx = int(np.round(dimx * max_size / max_dim_size))
                dimy = int(np.round(dimy * max_size / max_dim_size))
                dimz = int(np.round(dimz * max_size / max_dim_size))

            geometry_tensor = torch.from_numpy(geometry_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
            augmented_geometry_tensor = F.interpolate(geometry_tensor, size=(dimx, dimy, dimz), mode='nearest')  # using trilinear here has bug
            augmented_geometry_tensor = (augmented_geometry_tensor > 0.3).float()
            segmentation_tensor = torch.from_numpy(segmentation_vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
            augmented_segmentation_tensor = F.interpolate(segmentation_tensor, size=(dimx, dimy, dimz), mode='nearest')

            augmented_geometry = augmented_geometry_tensor.detach().cpu().numpy()[0, 0]
            augmented_segmentation = augmented_segmentation_tensor.detach().cpu().numpy()[0, 0]
            padx_left = (self.output_size - dimx) // 2
            pady_left = (self.output_size - dimy) // 2
            padz_left = (self.output_size - dimz) // 2
            padx_right = self.output_size - dimx - padx_left
            pady_right = self.output_size - dimy - pady_left
            padz_right = self.output_size - dimz - padz_left
            augmented_geometry = np.pad(augmented_geometry, ((padx_left, padx_right), (pady_left, pady_right), (padz_left, padz_right)), 'constant', constant_values=0)
            augmented_segmentation = np.pad(augmented_segmentation, ((padx_left, padx_right), (pady_left, pady_right), (padz_left, padz_right)), 'constant', constant_values=0)
            augmented_geo_org = np.round(augmented_geometry).astype(np.uint8)
            augmented_seg_org = np.round(augmented_segmentation).astype(np.uint8)

            if np.random.randint(2):
                rotation_degree = np.random.randint(4)
                augmented_geo_org = np.rot90(augmented_geo_org, k=rotation_degree, axes=(0, 2)).copy()
                augmented_seg_org = np.rot90(augmented_seg_org, k=rotation_degree, axes=(0, 2)).copy()

            return augmented_geo_org, augmented_seg_org

    def get_voxel_bbox(self, vox):
        # mini map
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallx, smally, smallz = smallmaskx.shape
        # x
        ray = np.max(smallmaskx, (1, 2))
        xmin = 0
        xmax = 0
        for i in range(smallx):
            if ray[i] > 0:
                if xmin == 0:
                    xmin = i
                xmax = i
        # y
        ray = np.max(smallmaskx, (0, 2))
        ymin = 0
        ymax = 0
        for i in range(smally):
            if ray[i] > 0:
                if ymin == 0:
                    ymin = i
                ymax = i
        # z
        ray = np.max(smallmaskx, (0, 1))
        zmin = 0
        zmax = 0
        for i in range(smallz):
            if ray[i] > 0:
                if zmin == 0:
                    zmin = i
                zmax = i

        return xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1

    def crop_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax, margin=True):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin

        if margin:
            tmp = np.zeros([xspan * self.upsample_rate + self.mask_margin * 2,
                            yspan * self.upsample_rate + self.mask_margin * 2,
                            zspan * self.upsample_rate + self.mask_margin * 2], np.uint8)

            tmp[self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin] = vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                                                          ymin * self.upsample_rate:ymax * self.upsample_rate,
                                                          zmin * self.upsample_rate:zmax * self.upsample_rate]
        else:
            tmp = vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                      ymin * self.upsample_rate:ymax * self.upsample_rate,
                      zmin * self.upsample_rate:zmax * self.upsample_rate]

        return tmp

    @staticmethod
    def recover_voxel_by_padding(vox, output_size):
        dimx, dimy, dimz = vox.shape

        output_size = max(dimx, dimy, dimz) if output_size < max(dimx, dimy, dimz) else output_size

        padx_left = (output_size - dimx) // 2
        pady_left = (output_size - dimy) // 2
        padz_left = (output_size - dimz) // 2
        padx_right = output_size - dimx - padx_left
        pady_right = output_size - dimy - pady_left
        padz_right = output_size - dimz - padz_left

        tmp = np.pad(vox, ((padx_left, padx_right), (pady_left, pady_right), (padz_left, padz_right)), 'constant', constant_values=0)
        assert tmp.shape == (output_size, output_size, output_size)

        return tmp

    def load(self):
        # load previous checkpoint
        model_path = os.path.join(self.checkpoint_dir, self.checkpoint_model)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.generator.load_state_dict(checkpoint['generator'])
            for i in range(self.upsample_level):
                self.discriminator[i].load_state_dict(checkpoint['discriminator_l' + str(i + 1)])
            print(" [*][{}] Load SUCCESS".format(model_path))
            return True
        else:
            print(" [!][{}] not exist".format(model_path))
            return False

    def save(self, epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path, 'IM_AE.model-' + str(epoch + 1) + ".pth")
        # save checkpoint
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator_l1': self.discriminator[0].state_dict(),
            'discriminator_l2': self.discriminator[1].state_dict(),
            'discriminator_l3': self.discriminator[2].state_dict(),
            # 'discriminator_l4': self.discriminator[3].state_dict(),
        }, save_dir)

    def print_param(self):
        print("-------------------------------------------")
        print("Number of parts:          ", self.num_parts)
        print("Generator # param:        ", sum(p.numel() for p in self.generator.parameters() if p.requires_grad))
        print("Discriminator l1 # param: ", sum(p.numel() for p in self.discriminator[0].parameters() if p.requires_grad))
        print("Discriminator l2 # param: ", sum(p.numel() for p in self.discriminator[1].parameters() if p.requires_grad))
        print("Discriminator l3 # param: ", sum(p.numel() for p in self.discriminator[2].parameters() if p.requires_grad))
        print("Alpha:                    ", self.param_alpha)
        print("Beta:                     ", self.param_beta)
        print("Input size:               ", self.input_size)
        print("Output size:              ", self.output_size)
        print("Upsample rate:            ", self.upsample_rate)
        print("Checkpoint dir:           ", self.checkpoint_path)
        print("Checkpoint model:         ", self.checkpoint_model)
        print("-------------------------------------------")

    def train(self, config):

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)
        iter_counter = 0

        for i in range(self.upsample_level):
            self.discriminator[i].train()
        self.generator.train()

        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)

            for idx in range(self.dataset_len):

                # reconstruction step
                r_steps = 4 if iter_counter < 5000 else 1
                iter_counter += 1
                for r_step in range(r_steps):
                    qxp = np.random.randint(self.styleset_len)

                    input_style = torch.from_numpy(self.input_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    Smask_style = torch.from_numpy(self.Smask_style[qxp]).to(self.device).unsqueeze(0).float()

                    # all parts share the same style
                    _, _, dimx, dimy, dimz = input_style.size()
                    tmp_Smask_style = Smask_style.view(self.num_parts, dimx * dimy * dimz)
                    z_tensor_recon = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    z_tensor_recon[qxp] = 1
                    zs_recon = torch.matmul(self.generator.style_codes, z_tensor_recon)  # (self.z_dim, self.num_parts)
                    zs_recon = torch.matmul(zs_recon, tmp_Smask_style).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    self.generator.zero_grad()

                    voxel_fake = self.generator(input_style, Smask_style * input_style, zs_recon, is_training=True)
                    # voxel_fake = self.generator(input_style, zs_recon, is_training=True)

                    loss_r = 0.
                    loss_r_log = []
                    for i in range(self.upsample_level):
                        voxel_style = torch.from_numpy(self.voxel_style[qxp][i]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        loss_r_i = torch.mean((voxel_style - voxel_fake[i]) ** 2) * self.param_beta
                        loss_r = loss_r + loss_r_i
                        loss_r_log.append(loss_r_i.item())

                    loss_r.backward()
                    self.optimizer_g.step()

                # G step
                g_steps = 1
                for g_step in range(g_steps):

                    self.generator.zero_grad()

                    dxb = batch_index_list[idx]
                    input_fake = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    Smask_fake = torch.from_numpy(self.Smask_content[dxb]).to(self.device).unsqueeze(0).float()

                    # random z vector for each part for D training
                    _, _, dimx, dimy, dimz = input_fake.size()
                    tmp_Smask_fake = Smask_fake.view(self.num_parts, dimx * dimy * dimz)
                    zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    random_style_idx = [random.sample(self.has_parts_style[i], k=1)[0] for i in range(self.num_parts)]
                    zs_vector[random_style_idx, torch.arange(self.num_parts)] = 1
                    zs = torch.matmul(self.generator.style_codes, zs_vector)  # (self.z_dim, self.num_parts)
                    zs = torch.matmul(zs, tmp_Smask_fake).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    voxel_fake = self.generator(input_fake, Smask_fake * input_fake, zs, is_training=True)
                    # voxel_fake = self.generator(input_fake, zs, is_training=True)
                    voxel_temp = (input_fake,) + voxel_fake[:-1]

                    loss_g = 0.
                    loss_a = 0.
                    loss_u = 0.
                    loss_g_log = []
                    loss_a_log = []
                    loss_u_log = []
                    for i in range(self.upsample_level):

                        Jmask_fake = torch.from_numpy(self.Jmask_content[dxb][i]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        Dmask_fake = torch.from_numpy(self.Dmask_content[dxb][i]).to(self.device).unsqueeze(0).float()
                        Dmask_fake_global = torch.max(Dmask_fake, dim=1, keepdim=True)[0]
                        Jmask_fake[Jmask_fake == 1] = 0.1
                        Jmask_fake[Jmask_fake == 0] = self.param_alpha

                        with torch.no_grad():
                            _, _, dimx, dimy, dimz = Dmask_fake.size()
                            Dmask_fake = Dmask_fake.view(self.num_parts, dimx * dimy * dimz)
                            Dmask_fake = torch.matmul(zs_vector, Dmask_fake).view(1, self.styleset_len, dimx, dimy, dimz)

                        D_out = self.discriminator[i](voxel_fake[i], is_training=False)
                        loss_g_i = (torch.sum((D_out[:, :-1] - 1) ** 2 * Dmask_fake * Jmask_fake) +
                                    torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_fake_global)) / torch.sum(Dmask_fake_global)
                        loss_g = loss_g + loss_g_i
                        loss_g_log.append(loss_g_i.item())

                        downsampled_avg_fake = F.avg_pool3d(voxel_fake[i], kernel_size=2, stride=2, padding=0)
                        downsampled_max_fake = F.max_pool3d(voxel_fake[i], kernel_size=2, stride=2, padding=0)

                        loss_a_i = torch.mean(torch.where(voxel_temp[i] > self.sampling_threshold,
                                                          torch.where(downsampled_max_fake > 0.6, torch.tensor(0.0).to(self.device), 1.0 - downsampled_avg_fake),
                                                          torch.where(downsampled_max_fake < 0.4, torch.tensor(0.0).to(self.device), downsampled_avg_fake)) ** 2)

                        loss_a = loss_a + loss_a_i
                        loss_a_log.append(loss_a_i.item())

                        upsampled_vox_temp = F.interpolate(voxel_temp[i], scale_factor=2, mode='nearest')
                        loss_u_i = torch.mean((upsampled_vox_temp - voxel_fake[i]) ** 2)

                        loss_u = loss_u + loss_u_i
                        loss_u_log.append(loss_u_i.item())

                    loss_gs = loss_g + loss_a * 10.0 + loss_u * 10.0
                    loss_gs.backward()
                    self.optimizer_g.step()

                # D step
                d_steps = 1
                for d_step in range(d_steps):

                    loss_d_real = 0.
                    loss_d_fake = 0.
                    loss_d_real_log = []
                    loss_d_fake_log = []
                    for i in range(self.upsample_level):

                        self.discriminator[i].zero_grad()

                        voxel_style = torch.from_numpy(self.voxel_style[qxp][i]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                        Dmask_style = torch.from_numpy(self.Dmask_style[qxp][i]).to(self.device).unsqueeze(0).float()
                        Dmask_fake = torch.from_numpy(self.Dmask_content[dxb][i]).to(self.device).unsqueeze(0).float()
                        Dmask_style_global = torch.max(Dmask_style, dim=1, keepdim=True)[0]
                        Dmask_fake_global = torch.max(Dmask_fake, dim=1, keepdim=True)[0]

                        with torch.no_grad():
                            _, _, dimx, dimy, dimz = Dmask_fake.size()
                            Dmask_fake = Dmask_fake.view(self.num_parts, dimx * dimy * dimz)
                            Dmask_fake = torch.matmul(zs_vector, Dmask_fake).view(1, self.styleset_len, dimx, dimy, dimz)

                            zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                            zs_vector[qxp] = 1
                            _, _, dimx, dimy, dimz = Dmask_style.size()
                            Dmask_style = Dmask_style.view(self.num_parts, dimx * dimy * dimz)
                            Dmask_style = torch.matmul(zs_vector, Dmask_style).view(1, self.styleset_len, dimx, dimy, dimz)

                        D_out = self.discriminator[i](voxel_style, is_training=True)
                        loss_d_real_i = (torch.sum((D_out[:, :-1] - 1) ** 2 * Dmask_style) +
                                         torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style_global)) / torch.sum(Dmask_style_global)
                        loss_d_real = loss_d_real + loss_d_real_i
                        loss_d_real_log.append(loss_d_real_i.item())

                        D_out = self.discriminator[i](voxel_fake[i].detach(), is_training=True)
                        loss_d_fake_i = (torch.sum((D_out[:, :-1]) ** 2 * Dmask_fake) +
                                         torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake_global)) / torch.sum(Dmask_fake_global)
                        loss_d_fake = loss_d_fake + loss_d_fake_i
                        loss_d_fake_log.append(loss_d_fake_i.item())

                    loss_d_real.backward()
                    loss_d_fake.backward()
                    self.optimizer_d.step()

                if epoch % 1 == 0 and (idx + 1) % (self.dataset_len // 2) == 0:
                    for k in range(self.upsample_level):
                        geometry_voxel_fake = voxel_fake[k].detach().cpu().numpy()[0, 0]
                        geometry_voxel = self.recover_voxel_by_padding(geometry_voxel_fake, self.output_size // 2 ** (self.upsample_level - k - 1))
                        vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                        vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                        file_name = '-'.join([str(i) for i in random_style_idx]) + '.ply'
                        write_ply_triangle(config.sample_dir + "/epoch" + str(epoch) + "_l" + str(k) + "_" + file_name, vertices, triangles)

            for i in range(self.upsample_level):
                print("Epoch: [%d/%d], time: %.0f, level: %d, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f, loss_a: %.6f, loss_u: %.6f" % (
                    epoch, training_epoch, time.time() - start_time, i, loss_d_real_log[i], loss_d_fake_log[i], loss_r_log[i], loss_g_log[i], loss_a_log[i], loss_u_log[i]))
            print()

            self.save(epoch)

        # if finished, save
        self.save(epoch)

    def test(self, config):
        # for chair-table
        if not self.load():
            exit(-1)

        self.generator.eval()

        data_dir = self.data_dir[:-1] + '_test'
        self.dataset_names = os.listdir(data_dir)
        self.dataset_len = len(self.dataset_names)

        print("testing {} contents with {} styles...".format(self.dataset_len, self.styleset_len))

        for i in range(self.dataset_len):

            voxel_path = os.path.join(data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            seg_path = os.path.join(data_dir, self.dataset_names[i] + "/segmentation_voxel.hdf5")
            data_dict = h5py.File(seg_path, 'r')
            seg_raw = data_dict["segmentation"][:]  # note that seg_vox is already (256, 256, 256)
            data_dict.close()

            if self.output_size == 128:
                geo_raw = get_vox_from_binvox_1over2_return_small(voxel_path).astype(np.uint8)
                seg_raw = self.get_downsampled_segmentation_voxel(seg_raw, downsample_levels=1)
            elif self.output_size == 256:
                geo_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(geo_raw)
            tmp_geo = self.crop_voxel(geo_raw, xmin, xmax, ymin, ymax, zmin, zmax)
            tmp_seg = self.crop_voxel(seg_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, tmp_Smask, _, _ = self.get_voxel_input_Smask_Dmasks_Jmasks(tmp_geo, tmp_seg)
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Smask_fake = torch.from_numpy(tmp_Smask).to(self.device).unsqueeze(0).float()

            save_dir = os.path.join(config.sample_dir, self.dataset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # else:
            #     continue

            test_id_sets = [[4, 11, 24, 8], [9, 6, 22, 1], [21, 4, 20, 10], [9, 11, 19, 5], [3, 3, 18, 9]]

            for test_id in test_id_sets:
                seat_id, arm_id, leg_id, back_id = test_id
                file_name = 'seat_{}_arm_{}_leg_{}_back_{}.ply'.format(seat_id, arm_id, leg_id, back_id)

                with torch.no_grad():
                    _, _, dimx, dimy, dimz = input_fake.size()
                    tmp_Smask_fake = Smask_fake.view(self.num_parts, dimx * dimy * dimz)
                    zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    zs_vector[[back_id, seat_id, leg_id, arm_id, 5], torch.arange(self.num_parts)] = 1
                    zs = torch.matmul(self.generator.style_codes, zs_vector)  # (self.z_dim, self.num_parts)
                    zs = torch.matmul(zs, tmp_Smask_fake).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    voxel_fake = self.generator(input_fake, Smask_fake * input_fake, zs, is_training=True)
                    # voxel_fake = self.generator(input_fake, zs, is_training=True)

                    geometry_voxel_fake = voxel_fake[-1].detach().cpu().numpy()[0, 0]
                    geometry_voxel = self.recover_voxel_by_padding(geometry_voxel_fake, self.output_size // 2 ** (self.upsample_level - 2 - 1))
                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    write_ply_triangle(save_dir + "/" + file_name, vertices, triangles)

            input_fake = F.interpolate(input_fake, scale_factor=self.upsample_rate, mode='nearest').detach().cpu().numpy()[0, 0]
            input_fake = self.recover_voxel_by_padding(input_fake, self.output_size)
            vertices, triangles = mcubes.marching_cubes(input_fake, self.sampling_threshold)
            vertices = (vertices + 0.5) / input_fake.shape[0] - 0.5
            write_ply_triangle(save_dir + "/" + "input_coarse_voxel.ply", vertices, triangles)

    def test_coarse_voxel_from_gui(self, config):
        # for chair-table
        if not self.load():
            exit(-1)

        self.generator.eval()

        data_dir = self.data_dir[:-1] + '_test'
        self.dataset_names = os.listdir(data_dir)
        self.dataset_len = len(self.dataset_names)

        print("testing {} contents with {} styles...".format(self.dataset_len, self.styleset_len))

        for i in range(self.dataset_len):

            data_path = os.path.join(data_dir, self.dataset_names[i] + "/gui_model_test.hdf5")
            data_dict = h5py.File(data_path, 'r')
            geo_raw = data_dict["coarse_geo"][:]  # (16, 16, 16)
            seg_raw = data_dict["coarse_seg"][:]  # (16, 16, 16)
            data_dict.close()

            xmin, xmax = np.nonzero(np.sum(geo_raw, axis=(1, 2)))[0][[0, -1]]
            ymin, ymax = np.nonzero(np.sum(geo_raw, axis=(0, 2)))[0][[0, -1]]
            zmin, zmax = np.nonzero(np.sum(geo_raw, axis=(0, 1)))[0][[0, -1]]
            print("xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}".format(xmin, xmax, ymin, ymax, zmin, zmax))
            tmp_geo = geo_raw[xmin - 1:xmax + 2, ymin - 2:ymax + 3, zmin - 2:zmax + 3]
            tmp_seg = seg_raw[xmin - 1:xmax + 2, ymin - 2:ymax + 3, zmin - 2:zmax + 3]

            input_fake = torch.from_numpy(tmp_geo).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Smask_fake = torch.from_numpy(tmp_seg).to(self.device).unsqueeze(0).unsqueeze(0).float()
            _, _, dimx, dimy, dimz = Smask_fake.size()
            Smask_fake = F.one_hot(Smask_fake.contiguous().view(dimx * dimy * dimz).long(),
                                   num_classes=self.num_parts + 1)[:, 1:].contiguous().permute(1, 0).view(1, self.num_parts, dimx, dimy, dimz).float()

            save_dir = os.path.join(config.sample_dir, self.dataset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            test_id_sets = [[4, 11, 24, 8], [9, 6, 22, 1], [21, 4, 20, 10], [9, 11, 19, 5], [3, 3, 18, 9]]

            for test_id in test_id_sets:
                seat_id, arm_id, leg_id, back_id = test_id
                file_name = 'seat_{}_arm_{}_leg_{}_back_{}.ply'.format(seat_id, arm_id, leg_id, back_id)
                with torch.no_grad():
                    _, _, dimx, dimy, dimz = input_fake.size()
                    tmp_Smask_fake = Smask_fake.view(self.num_parts, dimx * dimy * dimz)
                    zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    zs_vector[[back_id, seat_id, leg_id, arm_id, 5], torch.arange(self.num_parts)] = 1
                    zs = torch.matmul(self.generator.style_codes, zs_vector)  # (self.z_dim, self.num_parts)
                    zs = torch.matmul(zs, tmp_Smask_fake).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    voxel_fake = self.generator(input_fake, Smask_fake * input_fake, zs, is_training=True)
                    # voxel_fake = self.generator(input_fake, zs, is_training=True)

                    geometry_voxel_fake = voxel_fake[-1].detach().cpu().numpy()[0, 0]
                    geometry_voxel = self.recover_voxel_by_padding(geometry_voxel_fake, self.output_size // 2 ** (self.upsample_level - 2 - 1))
                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    write_ply_triangle(save_dir + "/" + file_name, vertices, triangles)

            input_fake = F.interpolate(input_fake, scale_factor=self.upsample_rate, mode='nearest').detach().cpu().numpy()[0, 0]
            input_fake = self.recover_voxel_by_padding(input_fake, self.output_size)
            vertices, triangles = mcubes.marching_cubes(input_fake, self.sampling_threshold)
            vertices = (vertices + 0.5) / input_fake.shape[0] - 0.5
            write_ply_triangle(save_dir + "/" + "input_coarse_voxel.ply", vertices, triangles)

    def test_coarse_voxel_from_gui_bpcc(self, config):

        if not self.load():
            exit(-1)

        self.generator.eval()

        data_dir = self.data_dir[:-1] + '_test'
        self.dataset_names = os.listdir(data_dir)
        self.dataset_len = len(self.dataset_names)

        print("testing {} contents with {} styles...".format(self.dataset_len, self.styleset_len))

        for i in range(self.dataset_len):

            data_path = os.path.join(data_dir, self.dataset_names[i] + "/gui_model_test.hdf5")
            data_dict = h5py.File(data_path, 'r')
            geo_raw = data_dict["coarse_geo"][:]  # (16, 16, 16)
            seg_raw = data_dict["coarse_seg"][:]  # (16, 16, 16)
            data_dict.close()

            mask_margin = 1
            xmin, xmax = np.nonzero(np.sum(geo_raw, axis=(1, 2)))[0][[0, -1]]
            ymin, ymax = np.nonzero(np.sum(geo_raw, axis=(0, 2)))[0][[0, -1]]
            zmin, zmax = np.nonzero(np.sum(geo_raw, axis=(0, 1)))[0][[0, -1]]
            print("xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}".format(xmin, xmax, ymin, ymax, zmin, zmax))
            tmp_geo = geo_raw[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]
            tmp_seg = seg_raw[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]
            tmp_geo = np.pad(tmp_geo, ((mask_margin, mask_margin), (mask_margin, mask_margin), (mask_margin, mask_margin)), 'constant', constant_values=0)
            tmp_seg = np.pad(tmp_seg, ((mask_margin, mask_margin), (mask_margin, mask_margin), (mask_margin, mask_margin)), 'constant', constant_values=0)

            labelled_points = np.stack((np.where(tmp_geo == 1))).T
            labels = tmp_seg[labelled_points[:, 0], labelled_points[:, 1], labelled_points[:, 2]]
            unlabelled_points = np.stack((np.where(tmp_geo == 0))).T
            labelled_tree = KDTree(labelled_points)
            _, inds = labelled_tree.query(unlabelled_points, k=1)
            tmp_seg[unlabelled_points[:, 0], unlabelled_points[:, 1], unlabelled_points[:, 2]] = labels[inds[:, 0]]

            input_fake = torch.from_numpy(tmp_geo).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Smask_fake = torch.from_numpy(tmp_seg).to(self.device).unsqueeze(0).unsqueeze(0).float()
            _, _, dimx, dimy, dimz = Smask_fake.size()
            Smask_fake = F.one_hot(Smask_fake.contiguous().view(dimx * dimy * dimz).long(),
                                   num_classes=self.num_parts + 1)[:, 1:].contiguous().permute(1, 0).view(1, self.num_parts, dimx, dimy, dimz).float()

            save_dir = os.path.join(config.sample_dir, self.dataset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            test_id_sets = [[10, 9]]

            for test_id in test_id_sets:
                bottom_id, top_id = test_id
                file_name = 'bottom_{}_top_{}.ply'.format(bottom_id, top_id)
                with torch.no_grad():
                    _, _, dimx, dimy, dimz = input_fake.size()
                    tmp_Smask_fake = Smask_fake.view(self.num_parts, dimx * dimy * dimz)
                    zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    zs_vector[[bottom_id, top_id], torch.arange(self.num_parts)] = 1
                    zs = torch.matmul(self.generator.style_codes, zs_vector)  # (self.z_dim, self.num_parts)
                    zs = torch.matmul(zs, tmp_Smask_fake).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    voxel_fake = self.generator(input_fake, Smask_fake * input_fake, zs, is_training=True)
                    # voxel_fake = self.generator(input_fake, zs, is_training=True)

                    geometry_voxel_fake = voxel_fake[-1].detach().cpu().numpy()[0, 0]
                    geometry_voxel = self.recover_voxel_by_padding(geometry_voxel_fake, self.output_size // 2 ** (self.upsample_level - 2 - 1))
                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    write_ply_triangle(save_dir + "/" + file_name, vertices, triangles)

            input_fake = F.interpolate(input_fake, scale_factor=self.upsample_rate, mode='nearest').detach().cpu().numpy()[0, 0]
            input_fake = self.recover_voxel_by_padding(input_fake, self.output_size)
            vertices, triangles = mcubes.marching_cubes(input_fake, self.sampling_threshold)
            vertices = (vertices + 0.5) / input_fake.shape[0] - 0.5
            write_ply_triangle(save_dir + "/" + "input_coarse_voxel.ply", vertices, triangles)

    def prepare_coarse_voxel(self, config):

        print("Prepare content coarse voxel for visualization...")
        for i in range(self.dataset_len):

            print(i, self.dataset_names[i])

            save_dir = os.path.join(config.sample_dir, self.dataset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            voxel_path = os.path.join(self.data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            vox_tensor = torch.from_numpy(tmp_raw).to(self.device).unsqueeze(0).unsqueeze(0).float()
            smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
            upsampled_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')
            upsampled = upsampled_tensor.detach().cpu().numpy()[0, 0]
            upsampled = np.round(upsampled).astype(np.uint8)

            vertices, triangles = mcubes.marching_cubes(upsampled, self.sampling_threshold)
            vertices = (vertices + 0.5) / upsampled.shape[0] - 0.5
            write_ply_triangle(save_dir + "/" + "geometry.ply", vertices, triangles)

    def prepare_segmented_coarse_voxel(self, config):

        color_maps = {1: [207, 244, 210], 2: [123, 228, 149], 3: [86, 197, 150], 4: [50, 157, 156], 5: [32, 80, 114]}

        data_dir = self.data_dir[:-1] + '_test'
        self.dataset_names = os.listdir(data_dir)
        self.dataset_len = len(self.dataset_names)

        print("preparing {} contents shapes...".format(self.dataset_len))

        for i in range(self.dataset_len):

            voxel_path = os.path.join(data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            seg_path = os.path.join(data_dir, self.dataset_names[i] + "/segmentation_voxel.hdf5")
            data_dict = h5py.File(seg_path, 'r')
            seg_raw = data_dict["segmentation"][:]  # note that seg_vox is already (256, 256, 256)
            data_dict.close()

            if self.output_size == 128:
                geo_raw = get_vox_from_binvox_1over2_return_small(voxel_path).astype(np.uint8)
                seg_raw = self.get_downsampled_segmentation_voxel(seg_raw, downsample_levels=1)
            elif self.output_size == 256:
                geo_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            vox_tensor = torch.from_numpy(geo_raw).to(self.device).unsqueeze(0).unsqueeze(0).float()
            smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
            upsampled_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')
            upsampled_coarse_geo = upsampled_tensor.detach().cpu().numpy()[0, 0]
            upsampled_coarse_geo = np.round(upsampled_coarse_geo).astype(np.uint8)

            smallseg = self.get_downsampled_segmentation_voxel(seg_raw, downsample_levels=self.upsample_level)
            smallseg_tensor = torch.from_numpy(smallseg).to(self.device).unsqueeze(0).unsqueeze(0).float()
            upsampled_tensor = F.interpolate(smallseg_tensor, scale_factor=self.upsample_rate, mode='nearest')
            upsampled_coarse_seg = upsampled_tensor.detach().cpu().numpy()[0, 0]
            upsampled_coarse_seg = np.round(upsampled_coarse_seg).astype(np.uint8)

            vertices, triangles = mcubes.marching_cubes(upsampled_coarse_geo, self.sampling_threshold)
            vertices_normalized = (vertices + 0.5) / upsampled_coarse_geo.shape[0] - 0.5
            colors = []
            for v in vertices:
                current_label = upsampled_coarse_seg[int(v[0]), int(v[1]), int(v[2])]
                colors.append(color_maps[current_label])
            colors = np.array(colors).astype(np.uint8)
            write_ply_triangle_color(data_dir + '/' + self.dataset_names[i] + "/color_geometry.ply", vertices_normalized, colors, triangles)

    def prepare_voxel_style(self, config):
        import binvox_rw_faster as binvox_rw
        # import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        fin = open("splits/" + self.data_style + ".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        for style_id in range(self.styleset_len):
            print("preprocessing style - " + str(style_id + 1) + "/" + str(self.styleset_len))
            vox_path = os.path.join(self.data_dir, self.styleset_names[style_id] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2_return_small(vox_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(vox_path).astype(np.uint8)
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            # tmp = gaussian_filter(tmp.astype(np.float32), sigma=1)
            # tmp = (tmp>self.sampling_threshold).astype(np.uint8)

            binvox_rw.write_voxel(tmp, result_dir + "/style_" + str(style_id) + ".binvox")
            # tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            # binvox_rw.write_voxel(tmp_input, result_dir+"/style_"+str(style_id)+"_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp, 0.5)
            # vertices = vertices-0.5
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+".ply", vertices, triangles)
            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+"_coarse.ply", vertices, triangles)

    def prepare_voxel_for_eval(self, config):
        import binvox_rw_faster as binvox_rw
        # import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        fin = open("splits/" + self.data_style + ".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        # load content shapes
        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))
            voxel_path = os.path.join(self.data_dir[:-1] + "_for_quan_test", self.dataset_names[content_id] + "/model_depth_fusion.binvox")
            seg_path = os.path.join(self.data_dir[:-1] + "_for_quan_test", self.dataset_names[content_id] + "/segmentation_voxel.hdf5")
            data_dict = h5py.File(seg_path, 'r')
            seg_raw = data_dict["segmentation"][:]  # note that seg_vox is already (256, 256, 256)
            data_dict.close()
            if self.output_size == 128:
                geo_raw = get_vox_from_binvox_1over2_return_small(voxel_path).astype(np.uint8)
                seg_raw = self.get_downsampled_segmentation_voxel(seg_raw, downsample_levels=1)
            elif self.output_size == 256:
                geo_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)

            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(geo_raw)
            tmp_geo = self.crop_voxel(geo_raw, xmin, xmax, ymin, ymax, zmin, zmax)
            tmp_seg = self.crop_voxel(seg_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, tmp_Smask, _, _ = self.get_voxel_input_Smask_Dmasks_Jmasks(tmp_geo, tmp_seg)

            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/content_"+str(content_id)+"_coarse.ply", vertices, triangles)

            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Smask_fake = torch.from_numpy(tmp_Smask).to(self.device).unsqueeze(0).float()

            random_ids = [[3, 6, 19, 14, 5], [6, 30, 7, 15, 5], [2, 24, 1, 11, 5], [6, 24, 19, 11, 5], [7, 30, 2, 6, 5], [4, 3, 20, 8, 5],
                          [9, 16, 23, 4, 5], [15, 30, 10, 8, 5], [13, 18, 2, 14, 5], [11, 31, 5, 14, 5], [14, 30, 31, 14, 5],
                          [14, 11, 19, 11, 5], [5, 9, 15, 6, 5], [8, 18, 26, 8, 5], [2, 17, 10, 14, 5], [11, 31, 26, 4, 5], [12, 0, 26, 11, 5],
                          [1, 22, 14, 4, 5], [10, 20, 2, 4, 5], [15, 12, 29, 14, 5], [2, 10, 30, 11, 5], [2, 27, 24, 6, 5], [2, 5, 4, 15, 5],
                          [13, 22, 10, 14, 5], [5, 13, 25, 8, 5], [0, 0, 4, 6, 5], [1, 26, 4, 15, 5], [9, 13, 20, 15, 5], [9, 1, 13, 11, 5],
                          [9, 10, 9, 6, 5], [6, 15, 10, 15, 5], [10, 0, 21, 8, 5]]

            for style_id in range(len(random_ids)):
                with torch.no_grad():
                    _, _, dimx, dimy, dimz = input_fake.size()
                    tmp_Smask_fake = Smask_fake.view(self.num_parts, dimx * dimy * dimz)
                    zs_vector = torch.zeros([self.styleset_len, self.num_parts], device=self.device)
                    zs_vector[random_ids[style_id], torch.arange(self.num_parts)] = 1
                    zs = torch.matmul(self.generator.style_codes, zs_vector)  # (self.z_dim, self.num_parts)
                    zs = torch.matmul(zs, tmp_Smask_fake).view(1, self.z_dim, dimx, dimy, dimz)  # (1, self.z_dim, dimx, dimy, dimz)

                    voxel_fake = self.generator(input_fake, Smask_fake * input_fake, zs, is_training=True)

                tmp_voxel_fake = voxel_fake[-1].detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake, result_dir + "/output_content_" + str(content_id) + "_style_" + str(style_id) + ".binvox")

                # vertices, triangles = mcubes.marching_cubes(tmp_voxel_fake, 0.5)
                # vertices = vertices-0.5
                # write_ply_triangle(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".ply", vertices, triangles)
