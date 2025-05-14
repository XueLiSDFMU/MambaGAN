from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F



class MambaLayer(nn.Module):
        def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
            super().__init__()
            self.dim = dim
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,

            )

        def forward(self, x):
            B, C = x.shape[:2]
            assert C == self.dim
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]
            x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)

            out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
            return out

class MambaEncoder(nn.Module):
     def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], out_indices=[0, 1, 2, 3]):
            super().__init__()

            self.downsample_layers = nn.ModuleList()
            stem = nn.Sequential(
                nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            )
            self.downsample_layers.append(stem)
            for i in range(3):
                downsample_layer = nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList()
            num_slices_list = [64, 32, 16, 8]
            cur = 0
            for i in range(4):
                stage = nn.Sequential(
                    *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])])
                self.stages.append(stage)
                cur += depths[i]

            self.out_indices = out_indices

            self.mlps = nn.ModuleList()
            for i_layer in range(4):
                layer = nn.InstanceNorm3d(dims[i_layer])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

        def forward_features(self, x):
            outs = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x)
                    x_out = self.mlps[i](x_out)
                    outs.append(x_out)

            return tuple(outs)

        def forward(self, x):
            x = self.forward_features(x)
            return x
