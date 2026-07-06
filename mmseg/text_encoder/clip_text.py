
from collections import OrderedDict
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from .utils.get_templates import get_predefined_templates
from .utils import tokenizer
from .utils.class_names import get_classes
from mmengine.runner.checkpoint import load_state_dict

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
                 out_idx=None,
                 prompt_length=0,
                 prompt_depth=0):
        super().__init__()
        self.layers = layers
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

@MODELS.register_module()
class CLIP_Text(BaseModule):

    def __init__(
            self,
            embed_dim: int,
            # text
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            templates: str = 'vild',
            cache_feature=False,
            frozen='frozen_all',
            init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.frozen = frozen
        if isinstance(templates, List):
            self.prompt_templates = templates
        else:
            self.prompt_templates = get_predefined_templates(templates)

        self.cache_feature = cache_feature
        if self.cache_feature:
            self.cache = {}
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
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

            checkpoint = state_dict.copy()
            for k, v in checkpoint.items():
                if 'visual' in k:
                    state_dict.pop(k)
            load_state_dict(self, state_dict, strict=False, logger=None)
            del checkpoint

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
        else:
            for name, params in self.named_parameters():
                params.requires_grad = True

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2).contiguous()
        x = self.transformer(x)
        x = x.permute(1, 0, 2).contiguous()
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]),text.argmax(dim=-1)] @ self.text_projection
        return x

    def template_encode(self, vocabulary):
        text_embed_bucket = []
        for template in self.prompt_templates:
            text_inputs = tokenizer.tokenize([template.format(noun) for noun in vocabulary]).to(self.device)
            text_embed = self.encode_text(text_inputs).float()
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_bucket.append(text_embed)
        text_embed = torch.stack(text_embed_bucket, dim=1)
        return text_embed

    def forward(self,cfg,num=None):
        if num==None:
            if cfg.dataset_name is None:
                dataset_name = None
            else:
                dataset_name = cfg.dataset_name.lower()
            vocabulary = cfg.vocabulary
        else:
            dataset_name = cfg.dataset_name[num].lower()
            vocabulary = cfg.vocabulary
        assert (dataset_name is not None or vocabulary is not None), \
            "text_encoder required either 'dataset_name' or 'vocabulary'"
        assert not (dataset_name is not None and vocabulary is not None), \
            "there is conflict between 'dataset_name' and 'vocabulary'"

        if dataset_name is None:
            class_names = vocabulary
            if self.cache_feature:
                new_classes = [word for word in class_names if word not in self.cache]
                if len(new_classes) > 0:
                    class_embeds = self.template_encode(new_classes)
                    self.cache.update(dict(zip(new_classes, class_embeds)))
                class_embeds = torch.stack([self.cache[word] for word in class_names])
            else:
                class_embeds = self.template_encode(class_names)

        else:
            class_names = get_classes(dataset_name)
            if self.cache_feature:
                if dataset_name not in self.cache:
                    class_embeds = self.template_encode(class_names)
                    self.cache[dataset_name] = class_embeds
                else:
                    class_embeds = self.cache[dataset_name]
            else:
                class_embeds = self.template_encode(class_names)
        return class_embeds
