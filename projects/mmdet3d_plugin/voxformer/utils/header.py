
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Header(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, input_dict,lidar_feats):

        ####原版
        # res = {}
        #
        # x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        #
        # x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        # #x3d_up_l1 = x3d_l1
        # _, feat_dim, w, l, h  = x3d_up_l1.shape
        #
        # x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)
        #
        # ssc_logit_full = self.mlp_head(x3d_up_l1)
        #
        # res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        #################
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]

        #x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        x3d_up_l1 = x3d_l1+lidar_feats
        _, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)
        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
        return res
