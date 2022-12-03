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
        self.local_align_model = TemporalConv(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            conv_type=3,
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
        framewise_feats = self.local_align_model(x.permute(0, 2, 1)).permute(0, 2, 1)
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
        self.local_align_model = TemporalConv(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            conv_type=3,
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
        framewise_feats = self.local_align_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.pos_emb
        x = self.long_term_model(x)
        contextual_feats = x
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
        pose_dim = 274
        self.proj = nn.Linear(pose_dim, pose_dim)
        self.short_term_model = TemporalConv(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            conv_type=3,
        )
        self.local_align_model = TemporalConv(
            input_size=hidden_dim,
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
        x = self.short_term_model(x.permute(0, 2, 1))
        framewise_feats = self.local_align_model(x).permute(0, 2, 1)
        x = x.permute(0, 2, 1) + self.pos_emb
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
            self.full_rgb_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//4, 768))
            self.full_rgb_seq_scale = nn.Parameter(torch.ones(1))

        if 'right_hand' in cue: 
            self.right_hand_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)
            self.right_hand_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//4, 768))
            self.right_hand_seq_scale = nn.Parameter(torch.ones(1))

        if 'left_hand' in cue: 
            self.left_hand_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)
            self.left_hand_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//4, 768))
            self.left_hand_seq_scale = nn.Parameter(torch.ones(1))

        if share_hand_model:
            self.right_hand_model = self.left_hand_model

        if 'face' in cue: 
            self.face_model = RGBCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)
            self.face_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//4, 768))
            self.face_seq_scale = nn.Parameter(torch.ones(1))

        if 'pose' in cue: 
            self.pose_model = PoseCueModel(
                num_classes=num_classes,
                hidden_dim=768,
                frame_len=frame_len,)
            self.pose_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//4, 768))
            self.pose_seq_scale = nn.Parameter(torch.ones(1))

        # self.local_seq = nn.Parameter(torch.randn(1, num_classes, frame_len//2, 768))
        
        glo_dim = 768*5
        self.pred_head = nn.Sequential(
            nn.Linear(glo_dim, glo_dim),
            nn.Linear(glo_dim, num_classes),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward_cue(self, x, cue):
        if cue != 'pose':
            x = x.to(self.device, non_blocking=True)
            model = eval(f'self.{cue}_model')
            return model(x)
        else:
            x = x.to(self.device, non_blocking=True)
            model = eval(f'self.{cue}_model')
            return model(x)

    def mutual_distill(self, ret, key):
        l = 0.0
        for x in self.cue:
            for y in self.cue:
                if x != y:
                    l = l - F.cosine_similarity(ret[x][key], ret[y][key], dim=-1).mean()
        return l

    def align_local_seq(self, ret, key):
        local_ret = {}
        for x in self.cue:
            local_seq = eval(f'self.{x}_seq')
            local_seq_scale = eval(f'self.{x}_seq_scale')
            logits = F.cosine_similarity(ret[x][key].unsqueeze(1), local_seq, dim=-1).mean(dim=-1)
            local_ret[f'local_align/{x}'] = {
                'logits': logits*local_seq_scale, 
                'scale': local_seq_scale,
                }
        return local_ret

    def forward(self, inputs):
        ret = {}
        feats_list = []
        for key in self.cue:
            value = inputs[key]
            logits, local_feats, contextual_feats, scale = self.forward_cue(value, key)
            ret[key] = {
                'logits': logits, 
                'local_feats': local_feats,
                'contextual_feats': contextual_feats,
                'scale': scale,
                }
            feats_list.append(contextual_feats[:, 0, :])
        feats = torch.cat(feats_list, dim=-1)
        ret['late_fusion'] = {
            'logits': self.pred_head(feats)*self.scale, 
            'scale': self.scale,
            }
        ret.update(self.align_local_seq(ret, 'local_feats'))
        # ret['mutual_distill_loss/framewise'] = self.mutual_dialign_local_seqstill(ret, 'framewise_feats')
        ret['mutual_distill_loss/contextual'] = self.mutual_distill(ret, 'contextual_feats')
        return ret
