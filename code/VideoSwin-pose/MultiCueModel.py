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
from video_swin_transformer import SwinTransformer3D
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
        contextual_feats = x[:, 0, :]
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
        contextual_feats = x[:, 0, :]
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, framewise_feats, contextual_feats, self.scale


class PoseCueModel(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        frame_len,
    ):
        super(PoseCueModel, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, frame_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.long_term_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pred_head = nn.Linear(hidden_dim, num_classes)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x[:, 0, :]
        logits = self.pred_head(x[:, 0, :])*self.scale
        B, N, C = x.shape
        framewise_feats = x.reshape(B, N//2, 2, C)[:, :, 0, :]
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

        if share_hand_model:
            self.right_hand_model = self.left_hand_model

        if 'face' in cue: 
            self.face_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)

        if 'pose' in cue: 
            pose_dim = 274
            self.pose_model = PoseCueModel(
                num_classes=num_classes,
                hidden_dim=pose_dim,
                frame_len=frame_len,)
        glo_dim = 768*4 + pose_dim
        self.pos_emb = nn.Parameter(torch.randn(1, frame_len//2, glo_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=glo_dim, nhead=7, batch_first=True)
        self.long_term_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pred_head = nn.Sequential(
            nn.Linear(glo_dim, glo_dim),
            nn.Linear(glo_dim, num_classes),
        )
        self.scale = nn.Parameter(torch.ones(1))
        
        glo_dim = glo_dim * 2
        self.glo_pred_head = nn.Sequential(
            nn.Linear(glo_dim, glo_dim),
            nn.Linear(glo_dim, num_classes),
        )
        self.glo_scale = nn.Parameter(torch.ones(1))

    def forward_cue(self, x, cue):
        if cue != 'pose':
            x = x.to(self.device, non_blocking=True)
            model = eval(f'self.{cue}_model')
            return model(x)
        else:
            x = x.to(self.device, non_blocking=True)
            model = eval(f'self.{cue}_model')
            return model(x)

    def forward(self, inputs):
        ret = {}
        framewise_feats_list = []
        contextual_feats_list = []
        for key, value in inputs.items():
            logits, framewise_feats, contextual_feats, scale = self.forward_cue(value, key)
            ret[key] = {
                'logits': logits, 
                'scale': scale,
                }
            framewise_feats_list.append(framewise_feats)
            contextual_feats_list.append(contextual_feats)
        x = torch.cat(framewise_feats_list, dim=-1)
        x = x + self.pos_emb
        x = self.long_term_model(x)
        logits = self.pred_head(x[:, 0, :])*self.scale
        ret['multi_cue'] = {
            'logits': logits, 
            'scale': self.scale,
            }

        contextual_feats_list.append(x[:, 0, :])
        x = torch.cat(contextual_feats_list, dim=-1)
        logits = self.glo_pred_head(x)*self.glo_scale
        ret['late_fusion'] = {
            'logits': logits, 
            'scale': self.glo_scale,
            }
        return ret
