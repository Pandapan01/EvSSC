
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import mmdet3d
from tkinter.messagebox import NO
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, builder
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.voxformer.modules.CBAM import ChannelAttention, SpatialAttention, CBAM
from projects.mmdet3d_plugin.voxformer.modules.CBAM import BACKBONES
from projects.mmdet3d_plugin.voxformer.modules.ELM import iAFF,AFF
from projects.mmdet3d_plugin.voxformer.modules.ELM import BACKBONES
from torch.nn import init


@DETECTORS.register_module()
class VoxFormerMultiModal_deep(MVXTwoStageDetector):
    """
    多模态输入版本的VoxFormer模型
    New args:
    --multimodal_backbone: 额外模态输入的backbone
    --mm_in_channels: 多模态的输入通道数
    --mm_fusion: 多模态的融合方式 选择为['add', 'cat', 'cat_1x1']
        --add: 相加融合
        --cat: 并联并扩大neck的input dimension融合
        --cat_1x1: 并联并使用1x1卷积压缩通道融合
    """

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 multimodal_backbone=None,
                 multimodal_fusion=None,
                 mm_in_channels=None,
                 mm_fusion=None,
                 pts_backbone=None,
                 img_neck=None,
                 ev_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):

        super(VoxFormerMultiModal_deep,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone,
                             img_neck,ev_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)  # 这里是重写MVXTwoStageDetector的init参数
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if multimodal_backbone is not None:
            assert mm_in_channels is not None
            self.mm_in_channels = int(mm_in_channels)
            self.mm_fusion = mm_fusion
            if mm_fusion is 'cat_1x1':
                self.mm_fusion_layer = nn.Sequential(
                    nn.Conv2d(1024 * 2, 1024, kernel_size=1, bias=False),
                    nn.BatchNorm2d(1024),
                )
            self.multimodal_backbone = builder.build_backbone(multimodal_backbone)
            print(BACKBONES.module_dict)
            if ev_neck is not None:
                self.ev_neck = builder.build_neck(ev_neck)

            self.multimodal_fusion = builder.build_backbone(multimodal_fusion)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            if self.multimodal_backbone:
                # 有多模态信息输入
                C = img.size()[1]
                img_img = img[:, :(C - self.mm_in_channels), ...]
                img_mm = img[:, -self.mm_in_channels:, ...]
                img_img_feats = self.img_backbone(img_img)
                img_mm_feats = self.multimodal_backbone(img_mm)
                residual1 = img_img_feats[0]
                residual2 = img_mm_feats[0]
                if self.mm_fusion is 'cat_1x1':
                    # 1x1 cat 融合
                    img_feats = (self.mm_fusion_layer(
                        torch.cat((img_img_feats[0], img_mm_feats[0]), dim=1)
                    ),)
                elif self.mm_fusion is 'cat':
                    # cat 融合
                    #print(img_img_feats[0].shape,img_mm_feats[0].shape,self.multimodal_fusion(img_mm_feats[0]).shape)
                    img_feats = (torch.cat((img_img_feats[0], img_mm_feats[0]), dim=1),)
                    #residual = (torch.cat((residual1,residual2),dim=1),)
                # elif self.mm_fusion is 'channel_attention':
                #     model=ChannelAttention(in_planes=2048,ratio=8)
                #     if torch.cuda.device_count() > 1:
                #         model = torch.nn.DataParallel(model)  # 使用多GPU
                #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #     model = model.to(device)
                #     img_feats = (model(torch.cat((img_img_feats[0], img_mm_feats[0]),dim=1)),)
                # elif self.mm_fusion is 'spatial_attention':
                #     model=SpatialAttention(in_planes=2048)
                #     img_feats = (model(torch.cat((img_img_feats[0], img_mm_feats[0]),dim=1)),)
                # elif self.mm_fusion is 'CBAM':
                #     model=CBAM(in_planes=2048,ratio=8)
                #     img_feats = (model(torch.cat((img_img_feats[0], img_mm_feats[0]),dim=1)),)
                else:
                    # 计算两个张量的和 创建一个新的元组，包含这两个张量的和
                    img_feats = (img_img_feats[0] + img_mm_feats[0],)  # 相加融合
            else:
                # 单图像输入
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            #img_feats = (self.multimodal_fusion(residual2,residual1),)
            img_feats = (residual1,)
            img_feats = self.img_neck(img_feats)
            ev_feats = (residual2,)
            ev_feats = self.ev_neck(ev_feats)

        img_feats_reshaped = []
        ev_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        for ev_feat in ev_feats:
            BN, C, H, W = ev_feat.size()
            if len_queue is not None:
                ev_feats_reshaped.append(ev_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                ev_feats_reshaped.append(ev_feat.view(B, int(BN / B), C, H, W))
        combined_feats = []
        for img_feat, ev_feat in zip(img_feats_reshaped, ev_feats_reshaped):
            # 确保形状相同
            assert img_feat.size() == ev_feat.size(), "Shape mismatch between img_feat and ev_feat"
            combined_feat = torch.cat((img_feat, ev_feat), dim=2)  # 在通道维度拼接
            combined_feats.append(combined_feat)
        return combined_feats

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          target):
        """Forward function'
        """
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        losses = self.pts_bbox_head.training_step(outs, target, img_metas)
        return losses

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      target=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue - 1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                     **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]

        img_metas = [each[len_queue - 1] for each in img_metas]
        img = img[:, -1, ...]
        img_feats = self.extract_feat(img=img)
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        return completion_results



