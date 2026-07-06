
from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module()
class FA_Neck(BaseModule):
    def __init__(self,
                 channels=768,
                 out_channels=[256,128,64],
                 up_kernel_size=[2,4,8],
                 up_stride=[2,4,8],
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample = nn.ModuleList()

        for i in range(len(self.out_channels)):
            self.upsample.append(nn.ConvTranspose2d(self.channels, self.out_channels[i], kernel_size=up_kernel_size[i], stride=up_stride[i]))

    def feature_map(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1)
        feature = torch.einsum('bchw, btc -> btchw', img_feats, text_feats)
        return feature


    def forward(self, clip_image, text_feats):
        clip_image = clip_image[::-1]
        img_proj = []
        B, M, C = clip_image[0].size()
        img_input = clip_image[0][:, 1:, :].permute(0, 2, 1).reshape(B, -1, int((M - 1) ** 0.5), int((M - 1) ** 0.5)).contiguous()

        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).float()
        text_feats = text_feats.repeat(B, 1, 1).float()
        for i in range(len(clip_image) - 1):
            img_inputs = clip_image[i+1][:, 1:, :].permute(0, 2, 1).reshape(B, -1, int((M - 1) ** 0.5), int((M - 1) ** 0.5)).contiguous()
            img_proj.append(self.upsample[i](img_inputs))

        early_fusion = self.feature_map(img_input, text_feats)

        sp_feature = early_fusion

        outputs = dict()
        outputs['sp_feature'] = sp_feature
        outputs['img_proj'] = img_proj
        outputs['text_feats'] = text_feats
        return outputs
