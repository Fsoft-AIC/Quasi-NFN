import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from typing import List, Tuple
from nfn.common import NetworkSpec, WeightSpaceFeatures
from nfn.layers.layer_utils import (
    set_init_,
    set_init_einsum_,
    shape_wsfeat_symmetry,
    unshape_wsfeat_symmetry,
)

from examples.basic_cnn.helpers import make_cnn, sample_perm_scale, check_perm_scale_symmetry
from torch.utils.data.dataloader import default_collate
from nfn.common import state_dict_to_tensors, WeightSpaceFeatures, network_spec_from_wsfeat
from nfn.layers.equiv_layers import *
from nfn.layers.mlp_quasi import * 

class HNPS_SirenLinearQuasi(nn.Module):
    def __init__(self, in_network_spec: NetworkSpec, out_network_spec: NetworkSpec, in_channels, out_channels, scale_type="per", init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.in_network_spec = in_network_spec
        self.out_network_spec = out_network_spec
        layer_weight_shapes = in_network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(in_network_spec)
        in_filter_facs = [int(np.prod(spec.shape[2:])) for spec in in_network_spec.weight_spec]
        out_filter_facs = [int(np.prod(spec.shape[2:])) for spec in out_network_spec.weight_spec]
        if scale_type=='all':
            self.quasi_alpha = GLQuasiAlpha_Pooled_All_Improved(in_network_spec, in_channels, out_channels, hidden_dim = 64, positive_transform = "sin", scale = 0.1)
        elif scale_type=="per":
            self.quasi_alpha = GLQuasiAlpha_Pooled_Improved(in_network_spec, in_channels, out_channels, hidden_dim = 32, positive_transform = "sin", scale = 0.1) #,hidden_dim=32, backbone_blocks=0, max_scale=100)
        else:
            self.quasi_alpha = GLQuasiAlpha_Pooled_Output(in_network_spec, in_channels, out_channels)
        self.scale_type = scale_type
        for i in range(L):
            in_filter_fac = in_filter_facs[i]
            out_filter_fac = out_filter_facs[i]
            if i == 0:
                self.layer_0_Y_W = EinsumLayer(equation="mnqk, bnjq -> bmjk",
                                              weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels,
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_Y_b = EinsumLayer(equation="mnk, bnj -> bmjk",
                                               weight_shape=[out_filter_fac * out_channels, in_channels,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])
                self.layer_0_z_W = EinsumLayer(equation="mnq, bnjq -> bmj",
                                               weight_shape=[out_channels, in_filter_fac * in_channels,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_z_b = EinsumLayer(equation="mn, bnj -> bmj",
                                               weight_shape=[out_channels, in_channels],
                                               fan_in_mask=[0, 1])

                set_init_einsum_(
                    self.layer_0_Y_W,
                    self.layer_0_Y_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_z_W,
                    self.layer_0_z_b,
                    init_type=init_type,
                )

            elif i == L-1:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mnpj, bnpk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                # New layer for SIREN
                self.add_module(f"layer_{i}_Y_b",
                                EinsumLayer(equation="mnj, bnk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_channels,
                                                          layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mnpj, bnp -> bmj",
                                            weight_shape=[out_channels, in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_tau",
                                EinsumLayer(equation="mj, b-> bmj",
                                            weight_shape=[out_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0])
                                )

                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_b"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    getattr(self, f"layer_{i}_z_tau"),
                    init_type=init_type,
                )

            elif i == L-2:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                # New layer for SIREN
                self.add_module(f"layer_{i}_z_W",
                                EinsumLayer(equation="mnp, bnpj -> bmj",
                                            weight_shape=[out_channels, in_filter_fac * in_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )

                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )

            else:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )

                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )


    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        if self.scale_type in ['all', 'per']:
            all_scale = self.quasi_alpha(wsfeat)

        wsfeat = shape_wsfeat_symmetry(wsfeat, self.in_network_spec)
        out_weights, out_biases = [], []
        L = len(self.in_network_spec)
        for i in range(L):
            weight, bias = wsfeat[i]
            if  i == 0:
                Y_W = self.layer_0_Y_W(weight)
                Y_b = self.layer_0_Y_b(bias)
                out_weights.append(Y_W + Y_b)

                z_W = self.layer_0_z_W(weight)
                z_b = self.layer_0_z_b(bias)
                out_biases.append(z_W + z_b)

            elif i == L-1:
                prev_bias = wsfeat[i-1][1]
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                Y_b = getattr(self, f"layer_{i}_Y_b")(prev_bias) # Previous bias
                out_weights.append(Y_W + Y_b)

                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                z_tau = getattr(self, f"layer_{i}_z_tau")(torch.tensor([1], device=weight.device))
                out_biases.append(z_b + z_tau)

            elif i == L-2:
                next_weight = wsfeat[i+1][0]
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)

                z_W = getattr(self, f"layer_{i}_z_W")(next_weight) # Next weight
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b + z_W)

            else:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)

                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b)

        equiv_wsfeat = unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.in_network_spec)
        if self.scale_type not in ['all', 'per']:
            all_scale = self.quasi_alpha(equiv_wsfeat)
        prev_perm = None
        prev_scale = None
        out_weights, out_biases = [], []
        if self.scale_type == "all":
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None: #and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]

                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, in_dim] -> [batch_size, 1,  1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = weight * prev_scale.unsqueeze(1).unsqueeze(1) ** (-1)
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    # weight = weight[:, :, perm]
                    
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(1).unsqueeze(1)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)
                    bias = bias * scale.unsqueeze(1)          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)


        elif self.scale_type == "per":
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None: # and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]

                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(2).unsqueeze(2).unsqueeze(2) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = (weight * prev_scale.unsqueeze(2) ** (-1))
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # weight = weight[:, :, perm]

                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)

                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2).unsqueeze(2).unsqueeze(2)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)

                    bias = bias * scale          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)
        else:
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None:# and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]
                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(2).unsqueeze(2).unsqueeze(2) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = (weight * prev_scale.unsqueeze(2) ** (-1))
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # weight = weight[:, :, perm]

                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2).unsqueeze(2).unsqueeze(2)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)

                    bias = bias * scale          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)


        return WeightSpaceFeatures(out_weights, out_biases)

class HNPSLinearQuasi(nn.Module):
    def __init__(self, in_network_spec: NetworkSpec, out_network_spec: NetworkSpec, in_channels, out_channels, scale_type = "per",init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.in_network_spec = in_network_spec
        self.out_network_spec = out_network_spec

        layer_weight_shapes = in_network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(in_network_spec)
        in_filter_facs = [int(np.prod(spec.shape[2:])) for spec in in_network_spec.weight_spec]
        out_filter_facs = [int(np.prod(spec.shape[2:])) for spec in out_network_spec.weight_spec]
        if scale_type=='all':
            # self.quasi_alpha = GLQuasiAlpha_Pooled_All_Improved(in_network_spec, in_channels, out_channels, hidden_dim = 64, positive_transform = "sin", scale = 0.1, use_global_channel_stats=True)
            self.quasi_alpha = GLQuasiAlpha_Pooled_Combined_All(in_network_spec, in_channels, out_channels, hidden_dim = 32, positive_transform = "sin", scale = 0.01, use_global_channel_stats=True)
        elif scale_type=="per":
            # self.quasi_alpha = GLQuasiAlpha_Pooled_Improved(in_network_spec, in_channels, out_channels, hidden_dim = 32, positive_transform = "sin", scale = 0.01, use_global_channel_stats=True) #,hidden_dim=32, backbone_blocks=0, max_scale=100)
            self.quasi_alpha = GLQuasiAlpha_Pooled_Combined_Per(in_network_spec, in_channels, out_channels, hidden_dim = 32, positive_transform = "sin", scale = 0.01, use_global_channel_stats=True) #,hidden_dim=32, backbone_blocks=0, max_scale=100)

        else:
            self.quasi_alpha = GLQuasiAlpha_Pooled_Output(in_network_spec, in_channels, out_channels)

        # self.permutation = SinkhornQuasi(in_network_spec, in_channels, out_channels, n_iters=20, temp=1.0)

        for i in range(L):
            in_filter_fac = in_filter_facs[i]
            out_filter_fac = out_filter_facs[i]
            if i == 0:
                self.layer_0_Y_W = EinsumLayer(equation="mnqk, bnjq -> bmjk", 
                                              weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels, 
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_Y_b = EinsumLayer(equation="mnk, bnj -> bmjk",
                                               weight_shape=[out_filter_fac * out_channels, in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])
                self.layer_0_z_W = EinsumLayer(equation="mnq, bnjq -> bmj",
                                               weight_shape=[out_channels, in_filter_fac * in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_z_b = EinsumLayer(equation="mn, bnj -> bmj",
                                               weight_shape=[out_channels, in_channels],
                                               fan_in_mask=[0, 1])

                set_init_einsum_(
                    self.layer_0_Y_W,
                    self.layer_0_Y_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_z_W,
                    self.layer_0_z_b,
                    init_type=init_type,
                )

            elif i == L-1:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mnpj, bnpk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mnpj, bnp -> bmj",
                                            weight_shape=[out_channels, in_filter_fac * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_tau",
                                EinsumLayer(equation="mj, b-> bmj",
                                            weight_shape=[out_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                attributes = []
                attributes.extend([getattr(self, f"layer_{i}_z_b")])
                attributes.extend([getattr(self, f"layer_{i}_z_tau")])

                set_init_einsum_(*attributes,
                    init_type=init_type,
                )                
            else:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )
        self.scale_type = scale_type
    
    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        if self.scale_type in ['all', 'per']:
            all_scale = self.quasi_alpha(wsfeat)
        # all_perm = self.permutation.get_gumbel_hard_permutations(wsfeat)
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.in_network_spec)
        out_weights, out_biases = [], []
        L = len(self.in_network_spec)
        for i in range(L):
            weight, bias = wsfeat[i]
            if  i == 0:
                Y_W = self.layer_0_Y_W(weight)
                Y_b = self.layer_0_Y_b(bias)
                #make a random tensor for Y_b with the same shape as Y_W
                #Y_b = torch.randn_like(Y_W)
                out_weights.append(Y_W + Y_b)
                
                z_W = self.layer_0_z_W(weight)
                z_b = self.layer_0_z_b(bias)
                out_biases.append(z_W + z_b)

            elif i == L-1:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                z_tau = getattr(self, f"layer_{i}_z_tau")(torch.tensor([1], device=weight.device))
                out_biases.append(z_b + z_tau)
            
            else:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b)
                
        
        equiv_wsfeat = unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.in_network_spec)
        if self.scale_type not in ['all', 'per']:
            all_scale = self.quasi_alpha(equiv_wsfeat)
        prev_perm = None
        prev_scale = None
        out_weights, out_biases = [], []
        if self.scale_type == "all":
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None: #and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]

                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, in_dim] -> [batch_size, 1,  1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = weight * prev_scale.unsqueeze(1).unsqueeze(1) ** (-1)
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    # weight = weight[:, :, perm]
                    
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(1).unsqueeze(1)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)
                    bias = bias * scale.unsqueeze(1)          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)


        elif self.scale_type == "per":
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None: # and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]

                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(2).unsqueeze(2).unsqueeze(2) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = (weight * prev_scale.unsqueeze(2) ** (-1))
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # weight = weight[:, :, perm]

                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)

                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2).unsqueeze(2).unsqueeze(2)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)

                    bias = bias * scale          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)
        else:
            # print(all_scale)
            for i in range(L):
                # print(f"Layer {i}:")
                weight, bias = equiv_wsfeat[i]
                # print("Before: ",weight.shape)
                if prev_scale is not None:# and prev_perm is not None:
                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(3, -1)-> [batch_size, channels, out_dim, kernel, kernel, in_dim]
                    # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, 1, 1, in_dim]
                    if len(weight.shape) == 6:
                        # weight = torch.einsum('bad njk,bnm->bad mjk', weight, prev_perm)
                        weight = (weight.transpose(3, -1) * prev_scale.unsqueeze(2).unsqueeze(2).unsqueeze(2) ** (-1)).transpose(3, -1)
                    else: 
                        # Linear layer weight shape: [batch_size, channels, out_dim, in_dim]
                        # Scale shape: [batch_size, channels, in_dim] -> [batch_size, channels, 1, in_dim]
                        # weight = torch.einsum('bad n,bnm->bad m', weight, prev_perm)

                        weight = (weight * prev_scale.unsqueeze(2) ** (-1))
                if i != L-1:
                    scale = all_scale[i]
                    # perm = all_perm[i]
                    # weight = weight[:, :, perm]

                    # Apply the same scale for all weight dimensions
                    # For weight: reshape scale to broadcast properly
                    # weight: [batch_size, channels, out_dim, in_dim, kernel, kernel] +  transpose(2, -1)-> [batch_size, channels, kernel, in_dim, kernel, out_dim]
                    # Scale shape: [batch_size, channels,  out_dim] -> [batch_size, channels, 1, 1, 1, out_dim]
                    try:
                        if len(weight.shape) == 6:
                            # weight = torch.einsum('ban djk,bnm->bam djk', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2).unsqueeze(2).unsqueeze(2)).transpose(2, -1)
                        else:
                            # weight = torch.einsum('ban d,bnm->bam d', weight, perm)
                            weight = (weight.transpose(2, -1) * scale.unsqueeze(2)).transpose(2, -1)
                    except:
                        print(weight.shape)
                        print(scale.shape)
                    # For bias: reshape scale to broadcast properly  
                    # Bias shape: [batch_size, channels, out_dim] 
                    # Scale shape: [batch_size, out_dim] -> [batch_size, 1, out_dim]
                    # print("Bias:", bias.shape)
                    # print("Scale:", scale.shape)
                    # bias = torch.einsum('ban,bnm->bam', bias, perm)

                    bias = bias * scale          
                    prev_scale = scale
                    # prev_perm = perm
                # print("After: ",weight.shape)
                out_weights.append(weight)
                out_biases.append(bias)


        return WeightSpaceFeatures(out_weights, out_biases)

    
    def test_equivariant(self, original_sd, permutated_sd):
        def make_cnn():
            """Define a small CNN whose weight space we will process with an NFN."""
            return nn.Sequential(
                nn.Conv2d(1, 16, 3), nn.ReLU(),
                nn.Conv2d(16, 16, 3), nn.ReLU(),
                nn.Conv2d(16, 16, 3), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, 10)
            )

        wts_and_bs, wts_and_bs_perm_scale = [], []
        for i in range(10):
            wts_and_bs.append(state_dict_to_tensors(original_sd[i]))
            wts_and_bs_perm_scale.append(state_dict_to_tensors(permutated_sd[i]))

        # Here we manually collate weights and biases (stack into batch dim).
        # When using a dataloader, the collate is done automatically.
        # default_collate output is [2 (weight and bias), num_layer, batch]
        wtfeat = WeightSpaceFeatures(*default_collate(wts_and_bs))
        wtfeat_perm = WeightSpaceFeatures(*default_collate(wts_and_bs_perm_scale))

        out = self.forward(wtfeat)
        out_of_perm = self.forward(wtfeat_perm)
        def check_params_eq(params1: WeightSpaceFeatures, params2: WeightSpaceFeatures):
            equal = True
            for p1, p2 in zip(params1.weights, params2.weights):
                print("original output:", p1[0][0][0][0])
                print("permuted output:", p2[0][0][0][0])
                print("Difference", p1[0][0][0][0]-p2[0][0][0][0])
                equal = equal and torch.allclose(p1, p2, atol=1e-1)
            for p1, p2 in zip(params1.biases, params2.biases):
                equal = equal and torch.allclose(p1, p2, atol=1e-1)
            return equal

        print(f"NFN is equivariant: {check_params_eq(out, out_of_perm)}.")
