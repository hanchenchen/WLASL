import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from module.video_swin_transformer import SwinTransformer3D
from module.tconv import TemporalConv
from einops import rearrange


class RGBCueModel(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        frame_len,
    ):
        super(RGBCueModel, self).__init__()
        self.short_term_model = SwinTransformer3D(
            pretrained='checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth',
            pretrained2d=False,
            patch_size=(2,4,4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8,7,7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True
        )
        self.short_term_model.init_weights('checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth')
        self.pos_emb = nn.Parameter(torch.randn(1, frame_len//2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.long_term_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pred_head = nn.Linear(hidden_dim, num_classes)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = rearrange(x, 'n c d h w -> n c d h w')
        x = self.short_term_model(x)
        x = rearrange(x, 'n d h w c -> n d (h w) c')
        x = x.mean(dim=2)
        framewise_feats = x
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, framewise_feats, contextual_feats, self.scale


class OpticalFlowCueModel(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        frame_len,
    ):
        super(RGBCueModel, self).__init__()
        self.short_term_model = SwinTransformer3D(
            pretrained2d=False,
            patch_size=(2,4,4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8,7,7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            in_chans=2,
        )
        self.short_term_model.init_weights()
        self.pos_emb = nn.Parameter(torch.randn(1, frame_len//2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.long_term_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pred_head = nn.Linear(hidden_dim, num_classes)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = rearrange(x, 'n c d h w -> n c d h w')
        x = self.short_term_model(x)
        x = rearrange(x, 'n d h w c -> n d (h w) c')
        x = x.mean(dim=2)
        framewise_feats = x
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, framewise_feats, contextual_feats, self.scale


class PoseCueModel(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dim,
        hidden_dim,
        frame_len,
    ):
        super(PoseCueModel, self).__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.short_term_model = TemporalConv(
            input_size=input_dim,
            hidden_size=hidden_dim,
            conv_type=3,
        )
        self.pos_emb = nn.Parameter(torch.randn(1, frame_len//2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.long_term_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pred_head = nn.Linear(hidden_dim, num_classes)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.proj(x)
        x = self.short_term_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        framewise_feats = x
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, framewise_feats, contextual_feats, self.scale


class MultiCueModel(nn.Module):
    def __init__(
        self,
        cue,
        num_classes,
        share_hand_model=True
    ):
        super(MultiCueModel, self).__init__()
        self.cue = cue
        self.num_classes = num_classes
        self.device = torch.device("cuda")
        frame_len = 32
        if 'full_rgb' in cue: 
            self.full_rgb_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)
            self.full_rgb_placeholder = nn.Parameter(torch.randn(1, 768))

        if 'right_hand' in cue: 
            self.right_hand_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)

        if 'left_hand' in cue: 
            self.left_hand_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)

        if 'face' in cue: 
            self.face_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)

        if 'frame_kpts' in cue: 
            self.frame_kpts_model = PoseCueModel(
                num_classes=num_classes,
                input_dim=25*2,
                hidden_dim=768,
                frame_len=frame_len,)
            
        if 'face_kpts' in cue: 
            self.face_kpts_model = PoseCueModel(
                num_classes=num_classes,
                input_dim=70*2,
                hidden_dim=768,
                frame_len=frame_len,)
            
        if 'right_hand_kpts' in cue: 
            self.right_hand_kpts_model = PoseCueModel(
                num_classes=num_classes,
                input_dim=21*2,
                hidden_dim=768,
                frame_len=frame_len,)
            self.pose_placeholder = nn.Parameter(torch.randn(1, 768))
            
        if 'left_hand_kpts' in cue: 
            self.left_hand_kpts_model = PoseCueModel(
                num_classes=num_classes,
                input_dim=21*2,
                hidden_dim=768,
                frame_len=frame_len,)
            self.pose_placeholder = nn.Parameter(torch.randn(1, 768))

        if share_hand_model:
            self.right_hand_model = self.left_hand_model
            self.right_hand_kpts_model = self.left_hand_kpts_model

        glo_dim = 768*8
        self.pred_head = nn.Sequential(
            nn.Linear(glo_dim, glo_dim),
            nn.Linear(glo_dim, num_classes),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward_cue(self, x, cue):
        x = x.to(self.device, non_blocking=True)
        model = eval(f'self.{cue}_model')
        return model(x)

    def mutual_distill(self, ret, key):
        l = 0.0
        for x in self.cue:
            for y in self.cue:
                if x != y:
                    l = l - F.cosine_similarity(ret[x][key], ret[y][key]).mean()
        return l

    def forward(self, inputs):
        ret = {}
        feats_list = []
        for key in self.cue:
            value = inputs[key]
            logits, framewise_feats, contextual_feats, scale = self.forward_cue(value, key)
            ret[key] = {
                'logits': logits, 
                'framewise_feats': framewise_feats,
                'contextual_feats': contextual_feats,
                'scale': scale,
                }
            feats_list.append(contextual_feats[:, 0, :])
        feats = torch.cat(feats_list, dim=-1)
        ret['late_fusion'] = {
            'logits': self.pred_head(feats)*self.scale, 
            'scale': self.scale,
            }
        # ret['mutual_distill_loss/framewise'] = self.mutual_distill(ret, 'framewise_feats')
        ret['mutual_distill_loss/contextual'] = self.mutual_distill(ret, 'contextual_feats')
        return ret
