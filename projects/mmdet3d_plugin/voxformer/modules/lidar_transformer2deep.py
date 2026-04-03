import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .deformable_self_attention import DeformSelfAttention
from .deformable_cross_attention import MSDeformableAttention3D
from mmdet.models import DETECTORS, builder


@TRANSFORMER.register_module()
class PerceptionTransformer2deep_lidar(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 multimodal_fusion1=None,
                 multimodal_fusion2 = None,
                 **kwargs):
        super(PerceptionTransformer2deep_lidar, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        self.multimodal_fusion1 = builder.build_backbone(multimodal_fusion1)
        self.multimodal_fusion2 = builder.build_backbone(multimodal_fusion2)
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, DeformSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats0,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain voxel features.
        """

        mlvl_feats = []
        mlvl_feats_ev = []
        mlvl_feats_lidar = []

        for combined_feat in mlvl_feats0:
            C_combined = combined_feat.size(2)  # 获取总通道数
            third_C = C_combined // 3  # 计算每一部分通道数

            # 第一部分通道
            mlvl_feat = combined_feat[:, :, :third_C, :, :]
            mlvl_feats.append(mlvl_feat)

            # 第二部分通道
            mlvl_feat_ev = combined_feat[:, :, third_C:2 * third_C, :, :]
            mlvl_feats_ev.append(mlvl_feat_ev)

            # 第三部分通道
            mlvl_feat_lidar = combined_feat[:, :, 2 * third_C:, :, :]
            mlvl_feats_lidar.append(mlvl_feat_lidar)
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #  #[N, 1, 64]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, 1, 64]

        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx[0], 3], :, :]
        unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx[0], 3], :, :]

        unmasked_ref_3d = torch.from_numpy(ref_3d[vox_coords[unmasked_idx[0], 3], :]) 
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device)
        
        # feat_flatten = []
        # spatial_shapes = []
        # for lvl, feat in enumerate(mlvl_feats):
        #     bs, num_cam, c, h, w = feat.shape
        #     spatial_shape = (h, w)
        #     feat = feat.flatten(3).permute(1, 0, 3, 2)
        #     if self.use_cams_embeds:
        #         feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
        #     feat = feat + self.level_embeds[None,
        #                                     None, lvl:lvl + 1, :].to(feat.dtype)
        #     spatial_shapes.append(spatial_shape)
        #     feat_flatten.append(feat)
        #
        # feat_flatten = torch.cat(feat_flatten, 2)
        # feat_flatten_ev = []
        # for lvl, feat_ev in enumerate(mlvl_feats_ev):
        #     feat_ev = feat_ev.flatten(3).permute(1, 0, 3, 2)
        #     if self.use_cams_embeds:
        #         feat_ev = feat_ev + self.cams_embeds[:, None, None, :].to(feat_ev.dtype)
        #     feat_ev = feat_ev+ self.level_embeds[None,
        #                                     None, lvl:lvl + 1, :].to(feat_ev.dtype)
        #     feat_flatten_ev.append(feat_ev)
        #
        # feat_flatten_ev = torch.cat(feat_flatten_ev, 2)
        #
        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        #
        # level_start_index = torch.cat((spatial_shapes.new_zeros(
        #     (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #
        # feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        # feat_flatten_ev = feat_flatten_ev.permute(0,2,1,3)
        # feat_flatten_com = feat_flatten + feat_flatten_ev
        #feat_flatten_com = torch.cat((feat_flatten, feat_flatten_ev), dim=3)
        # (num_cam, H*W, bs, embed_dims)
        feat_flatten = []
        feat_flatten_com = []
        spatial_shapes = []
        feat_flatten_ev = []
        feat_flatten_lidar = []
        for lvl, (feat, feat_ev , feat_lidar) in enumerate(zip(mlvl_feats, mlvl_feats_ev,mlvl_feats_lidar)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.squeeze(0)
            feat_ev = feat_ev.squeeze(0)
            feat_lidar = feat_lidar.squeeze(0)
            feat_add = feat_ev+feat
            #feat_com = self.multimodal_fusion(feat_add)
            feat_com1 = self.multimodal_fusion1(feat_ev,feat) #aff
            feat_com = self.multimodal_fusion1(feat_lidar, feat_com1)  # aff
            # feat_com = self.multimodal_fusion(feat, feat_ev)
            feat = feat.unsqueeze(0)
            feat_ev = feat_ev.unsqueeze(0)
            feat_com = feat_com.unsqueeze(0)
            # 处理 feat
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat_ev = feat_ev.flatten(3).permute(1, 0, 3, 2)
            feat_com = feat_com.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat_ev = feat_ev + self.cams_embeds[:, None, None, :].to(feat_ev.dtype)
                feat_com = feat_com + self.cams_embeds[:, None, None, :].to(feat_com.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

            feat_ev = feat_ev + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat_ev.dtype)
            feat_com = feat_com + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat_com.dtype)
            feat_flatten_ev.append(feat_ev)
            feat_flatten_com.append(feat_com)

        # 最后拼接
        feat_flatten = torch.cat(feat_flatten, 2)
        feat_flatten_ev = torch.cat(feat_flatten_ev, 2)
        feat_flatten_com = torch.cat(feat_flatten_com, 2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        feat_flatten_ev = feat_flatten_ev.permute(0,2,1,3)
        feat_flatten_com = feat_flatten_com.permute(0, 2, 1, 3)

        # TODO：在此之前进行深融合
        bev_embed = self.encoder(
            unmasked_bev_queries,
            feat_flatten_com,
            feat_flatten_com,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=unmasked_bev_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=None,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def diffuse_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        diffuse voxel features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) 
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        unmasked_ref_3d = torch.from_numpy(ref_3d[vox_coords[unmasked_idx[0], 3], :]) 
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(bev_queries.device)
        
        bev_embed = self.encoder(
            bev_queries,
            None,
            None,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=None,
            level_start_index=None,
            prev_bev=None,
            shift=None,
            **kwargs
        ) 
        
        return bev_embed
