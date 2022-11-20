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
        self.pred_head = nn.Linear(768, num_classes)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x.cuda()
        x = rearrange(x, 'n c d h w -> n c d h w')
        x = self.short_term_model(x)
        x = rearrange(x, 'n d h w c -> n d (h w) c')
        x = x.mean(dim=2)
        framewise_feats = x
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, self.scale.item()


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
        x = x.cuda()
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
        logits = self.pred_head(x[:, 0, :])*self.scale
        return logits, self.scale.item()


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
            self.right_hand_model.pos_emb = self.left_hand_model.pos_emb
            self.right_hand_model.short_term_model = self.left_hand_model.short_term_model
            self.right_hand_model.long_term_model = self.left_hand_model.long_term_model

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

    def forward_cue(self, x, cue):
        if cue != 'pose':
            model = eval(f'self.{cue}_model')
            return model(x)
        else:
            model = eval(f'self.{cue}_model')
            return model(x)

    def forward(self, inputs):
        ret = {}
        for key, value in inputs.items():
            logits, scale = self.forward_cue(value, key)
            ret[key] = {'logits': logits, 'scale': scale}
        return ret
