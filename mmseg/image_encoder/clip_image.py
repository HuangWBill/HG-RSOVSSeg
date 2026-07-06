import math
from collections import OrderedDict
from typing import Tuple, Union
from mmseg.registry import MODELS
from mmengine.logging import print_log
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.runner.checkpoint import load_state_dict
from mmengine.model import BaseModule
from mmseg.models.utils import resize

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1).contiguous()
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 out_origin=False,
                 out_idx=None,
                 use_proj=False,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.out_origin = out_origin
        self.out_idx = out_idx
        self.use_proj = use_proj
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        if self.use_proj ==True:
            embed_dim = width * 32
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                            heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x_o =[]
        if self.out_origin:
            x_o.append(x)
        if self.out_idx is not None:
            x = self.layer1(x)
            if 1 in self.out_idx:
                x_o.append(x)
            x = self.layer2(x)
            if 2 in self.out_idx:
                x_o.append(x)
            x = self.layer3(x)
            if 3 in self.out_idx:
                x_o.append(x)
            x = self.layer4(x)
            if self.use_proj ==True:
                x = self.attnpool(x)
            if 4 in self.out_idx:
                x_o.append(x)
            x=x_o
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.use_proj == True:
                x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_dense(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        L, N, D = y.shape  # L N 3D

        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0,
                                               3).reshape(3 * N, L, D // 3).contiguous()
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)

        q, k, v = y.tensor_split(3, dim=0)
        v = v.transpose(1, 0).contiguous() + x  # L N D

        v = v + self.mlp(self.ln_2(v))
        return v


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 out_origin=False,
                 out_idx=None,
                 prompt_length=0,
                 prompt_depth=0):
        super().__init__()
        self.layers = layers
        self.out_origin = out_origin
        self.out_idx = out_idx
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

        self.prompt_length = prompt_length
        self.prompt_depth = prompt_depth
        self.prompt_tokens = nn.Parameter(
            torch.zeros(prompt_depth, prompt_length,
                        width)) if prompt_length > 0 else None
        if self.prompt_tokens is not None:
            nn.init.xavier_uniform_(self.prompt_tokens)

    def forward(self, x: torch.Tensor, dense=False):
        x_o = []
        if self.out_origin:
            x_o.append(x)

        for i, resblock in enumerate(self.resblocks):
            if self.prompt_length > 0 and i < self.prompt_depth:
                length = self.prompt_length + 1 if i > 0 else 1
                x = torch.cat((x[0:1, :, :], self.prompt_tokens[i].repeat(
                    x.shape[1], 1, 1).permute(1, 0, 2).contiguous(), x[length:, :, :]))

            if i == self.layers - 1 and dense:
                x = resblock.forward_dense(x)
                x = torch.cat((x[0:1, :, :], x[self.prompt_length + 1::, :]),
                              dim=0)
            else:
                x = resblock(x)
            if self.out_idx is not None:
                if i in self.out_idx:
                    x_o.append(x)
        if self.out_idx is not None:
            x = x_o
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int, out_origin,out_idx,use_proj, prompt_depth: int,
                 prompt_length: int,interpolate_mode='bicubic'):
        super().__init__()
        self.out_idx =out_idx
        self.img_size = input_resolution
        self.interpolate_mode = interpolate_mode

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            out_origin=out_origin,
            out_idx=out_idx,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length)

        self.ln_post = LayerNorm(width)
        if use_proj == True:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None
        self.patch_size = patch_size
        self.input_resolution = input_resolution

    def init_weights(self,state_dict):
        if 'positional_embedding' in state_dict.keys():
            if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["positional_embedding"].shape} to '
                              f'{self.positional_embedding.shape}')
                h = w = self.img_size
                pos_size = int(
                    math.sqrt(state_dict['positional_embedding'].shape[0] - 1))
                state_dict['positional_embedding'] = self.resize_pos_embed(
                    state_dict['positional_embedding'],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), self.interpolate_mode)

        load_state_dict(self, state_dict, strict=False, logger=None)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[0]
        pos_embed_weight = pos_embed[1:]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[1]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).squeeze(0)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1).transpose(1,0)
        return pos_embed

    def forward(self, x: torch.Tensor, dense=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1],-1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)

        if dense and (x.shape[1] != self.positional_embedding.shape[0]):
            x = x + self.resized_pos_embed(self.input_resolution,
                                           x.shape[1]).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2).contiguous()
        x = self.transformer(x, dense)

        if self.out_idx is not None:
            for i in range(len(x)):
                x[i] = x[i].permute(1, 0, 2).contiguous()
            if dense:
                x[-1] = self.ln_post(x[-1][:, :, :])
            else:
                x[-1] = self.ln_post(x[-1][:, 0, :])

            if self.proj is not None:
                x[-1] = x[-1] @ self.proj
        else:
            x = x.permute(1, 0, 2).contiguous()
            if dense:
                x = self.ln_post(x[:, :, :])
            else:
                x = self.ln_post(x[:, 0, :])

            if self.proj is not None:
                x = x @ self.proj

        return x

    def resized_pos_embed(self, in_res, tgt_res, mode='bicubic'):
        L, D = self.positional_embedding.shape

        in_side = in_res // self.patch_size
        tgt_side = int((tgt_res - 1)**0.5)

        cls_pos = self.positional_embedding[0].unsqueeze(0)
        pos_embed = self.positional_embedding[1:].reshape(
            1, in_side, in_side, D).permute(0, 3, 1, 2).contiguous()
        resized_pos_embed = F.interpolate(
            pos_embed,
            size=(tgt_side, tgt_side),
            mode=mode,
            align_corners=False,
        )
        resized_pos_embed = resized_pos_embed.squeeze(0).reshape(
            D, -1).T

        return torch.cat((cls_pos, resized_pos_embed), dim=0)

@MODELS.register_module()
class CLIP_Image(BaseModule):
    def __init__(
        self,
        embed_dim: int,
        out_idx,
        use_proj,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_heads: int,
        vision_patch_size: int,
        out_origin=False,
        prompt_depth: int = 0,
        prompt_length: int = 0,
        frozen=None,
        interpolate_mode='bicubic',
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.frozen=frozen
        self.image_resolution = image_resolution

        if isinstance(vision_layers, (tuple, list)):
            assert prompt_length == 0 and prompt_depth == 0
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                out_origin=out_origin,
                out_idx=out_idx,
                use_proj=use_proj,
                input_resolution=image_resolution,
                width=vision_width)
        else:
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                out_origin=out_origin,
                out_idx=out_idx,
                use_proj=use_proj,
                prompt_depth=prompt_depth,
                prompt_length=prompt_length,
                interpolate_mode=interpolate_mode,
            )
        self._freeze()

    def init_weights(self):
        if isinstance(self.init_cfg, dict) and self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = self.init_cfg['checkpoint']
            if isinstance(checkpoint, torch.jit.RecursiveScriptModule):
                state_dict = checkpoint.state_dict()
            else:
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

            new_ckpt = OrderedDict()
            for k, v in state_dict.items():
                key_list = k.split('.')
                if key_list[0] == 'visual':
                    new_name = '.'.join(key_list[1:])
                    new_ckpt[new_name] = v

            self.visual.init_weights(new_ckpt)

    def _freeze(self):
        if self.frozen == 'frozen_all':
            for name, params in self.named_parameters():
                params.requires_grad = False
        elif self.frozen == 'attention':
            for name, params in self.named_parameters():
                if 'attn' in name or 'position' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        elif self.frozen == 'position':
            for name, params in self.named_parameters():
                if 'position' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        elif self.frozen is not None:
            for name, params in self.named_parameters():
                if self.frozen in name.split('.'):
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        else:
            for name, params in self.named_parameters():
                params.requires_grad = True


    def encode_image(self, image, masks=None, pool_mask=None, dense=False):
        if pool_mask is not None:
            return self.visual(image, mask=pool_mask, dense=dense)
        if masks is None:
            return self.visual(image, dense=dense)
        else:
            return self.visual(image, masks)


    def forward(self, inputs,dense=True):
        inputs = F.interpolate(
            inputs,
            size=self.image_resolution,
            mode='bilinear',
            align_corners=False)
        image_features = self.encode_image(inputs,dense=dense)

        return image_features
