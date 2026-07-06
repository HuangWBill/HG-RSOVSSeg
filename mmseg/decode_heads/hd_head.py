
import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels - guidance_channels,
            kernel_size=2,
            stride=2)
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            norm_cfg=dict(type='GN', num_groups=out_channels // 16))
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            norm_cfg=dict(type='GN', num_groups=out_channels // 16))

    def forward(self, x, guidance=None):
        x = self.up(x)

        if len(guidance.shape) != len(x.shape):
            T = x.size(0) // guidance.size(0)
            guidance = guidance.repeat(T, 1, 1, 1)
            x = torch.cat([x, guidance], dim=1)
        else:
            x = torch.cat([x, guidance], dim=1)
        x = self.conv1(x)
        return self.conv2(x)

class TAM(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels,  dropout=0.0):
        super(TAM, self).__init__()
        self.key_channels=key_channels
        self.v_in_channels=v_in_channels
        self.vis_project = nn.Sequential(nn.Conv1d(v_in_channels, v_in_channels, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.f_query = nn.Sequential(
            nn.Conv1d(v_in_channels, key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(key_channels),
        )

        self.f_value = nn.Sequential(
            nn.Conv1d(l_in_channels, v_in_channels, kernel_size=1, stride=1),
        )

        self.f_key = nn.Sequential(
            nn.Conv1d(l_in_channels, key_channels, kernel_size=1, stride=1),
        )

        self.W = nn.Sequential(
                    nn.Conv1d(v_in_channels, v_in_channels, kernel_size=1, stride=1),
                    nn.InstanceNorm1d(v_in_channels),
                )

        self.project_mm = nn.Sequential(nn.Conv1d(v_in_channels, v_in_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l):
        B,C,H,W=x.size()
        x=x.reshape(B,C,-1)
        l=l.permute(0, 2, 1).contiguous()
        vis = self.vis_project(x)
        v_query = self.f_query(x)
        v_query = v_query.permute(0, 2, 1).contiguous()
        l_key = self.f_key(l)
        l_value = self.f_value(l)
        v_query = v_query.reshape(B, H * W, self.key_channels)
        l_key = l_key.reshape(B, self.key_channels, -1)
        l_value = l_value.reshape(B, self.v_in_channels, -1)
        vl = torch.einsum('blc, bcn -> bln', v_query, l_key)
        vl = F.softmax(vl, dim=-1)
        g_vl = torch.einsum('bln, bcn -> bcln', vl, l_value)
        g_vl1 = self.W(g_vl.permute(0, 3, 1, 2).flatten(0,1)).contiguous()
        mm = torch.einsum('bcl, bncl -> bncl', vis, g_vl1.reshape(B,-1,self.v_in_channels,H*W))
        mm = self.project_mm(mm.flatten(0, 1).contiguous())
        mm = mm.reshape(-1,self.v_in_channels, H,W)
        return mm


@MODELS.register_module()
class HD_Head(BaseDecodeHead):
    def __init__(self,
                 key_channels=128,
                 decoder_dims=[64, 32, 16],
                 dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder_dims=decoder_dims
        self.TA = nn.ModuleList()
        self.updecoder = nn.ModuleList()
        for i in range(len(decoder_dims)):
            self.TA.append(TAM(decoder_dims[i], self.in_channels, key_channels, dropout=dropout))
            self.updecoder.append(UpBlock(self.channels if i == 0 else decoder_dims[i-1], decoder_dims[i], decoder_dims[i]))
        self.conv_seg = nn.Conv2d(decoder_dims[-1], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        img_proj=inputs['img_proj']
        text_feats=inputs['text_feats']
        feature=inputs['sp_feature']
        B=text_feats.shape[0]
        ta_feature1=feature.flatten(0, 1).contiguous()
        for i in range(len(self.decoder_dims)):
            ta_feature = self.TA[i](img_proj[i],text_feats)
            ta_feature1 = self.updecoder[i](ta_feature1, ta_feature)
        output = self.cls_seg(ta_feature1)
        H_ori, W_ori = output.shape[-2:]
        output = output.reshape(B, -1, H_ori, W_ori)
        return output
