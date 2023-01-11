import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(
        self, input_size, hidden_size, conv_type=2
    ):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ["K3"]
        elif self.conv_type == 1:
            self.kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ["K5", "P2", "K5", "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ["K5", "P2", "K5"]
        elif self.conv_type == 4:
            self.kernel_size = ["K3", "P2", "K3"]
        elif self.conv_type == 5:
            self.kernel_size = ["K5", "P2", "K5", "P2"]
        elif self.conv_type == 6:
            self.kernel_size = ["K3", "P2", "K3", "P2"]
        print(self.kernel_size)

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=int(ks[1])//2,
                    )
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, frame_feat):
        visual_feat = self.temporal_conv(frame_feat) # B C T
        return visual_feat
