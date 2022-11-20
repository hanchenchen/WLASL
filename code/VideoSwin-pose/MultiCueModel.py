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

class MultiCueModel(nn.Module):
    def __init__(
        self,
        cue,
        num_classes,
    ):
        super(MultiCueModel, self).__init__()
        self.cue = cue
        self.num_classes = num_classes
        if 'full_rgb' in cue: 
            self.full_rgb_model = SwinTransformer3D(
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
            self.full_rgb_model.init_weights('checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth')
            self.full_rgb_model.pos_emb = nn.Parameter(torch.randn(1, 16, 768))
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
            self.full_rgb_model.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.full_rgb_model.pred_head = nn.Linear(768, num_classes)
            self.full_rgb_model.scale = nn.Parameter(torch.ones(1))

        if 'right_hand' in cue or 'left_hand' in cue: 
            self.hand_model = SwinTransformer3D(
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
            self.hand_model.init_weights('checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth')
            self.hand_model.pos_emb = nn.Parameter(torch.randn(1, 16, 768))
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
            self.hand_model.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.hand_model.pred_head = nn.Linear(768, num_classes)
            self.hand_model.scale = nn.Parameter(torch.ones(1))
            self.right_hand_model = self.hand_model
            self.left_hand_model = self.hand_model

        if 'face' in cue: 
            self.face_model = SwinTransformer3D(
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
            self.face_model.init_weights('checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth')
            self.face_model.pos_emb = nn.Parameter(torch.randn(1, 16, 768))
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
            self.face_model.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.face_model.pred_head = nn.Linear(768, num_classes)
            self.face_model.scale = nn.Parameter(torch.ones(1))

        if 'pose' in cue: 
            pose_dim = 274
            encoder_layer = nn.TransformerEncoderLayer(d_model=pose_dim, nhead=2, batch_first=True)
            self.pose_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.pose_model.pos_emb = nn.Parameter(torch.randn(1, 32, pose_dim))
            self.pose_model.pred_head = nn.Linear(pose_dim, num_classes)
            self.pose_model.scale = nn.Parameter(torch.ones(1))


    def forward_cue(self, x, cue):
        if cue != 'pose':
            model = eval(f'self.{cue}_model')
            # embed frames by resnet
            x = x.cuda()
            x = rearrange(x, 'n c d h w -> n c d h w')
            x = model(x)
            x = rearrange(x, 'n d h w c -> n d (h w) c')
            x = x.mean(dim=2)
            framewise_feats = x
            x = x + model.pos_emb
            x = model.temporal_model(x)
            contextual_feats = x
            logits = model.pred_head(x[:, 0, :])*model.scale
            return logits, model.scale.item()
        else:
            model = eval(f'self.{cue}_model')
            x = x.cuda()
            x = x + model.pos_emb
            x = model(x)
            contextual_feats = x
            logits = model.pred_head(x[:, 0, :])*model.scale
            return logits, model.scale.item()

    def forward(self, inputs):
        ret = {}
        for key, value in inputs.items():
            logits, scale = self.forward_cue(value, key)
            ret[key] = {'logits': logits, 'scale': scale}
        return ret
