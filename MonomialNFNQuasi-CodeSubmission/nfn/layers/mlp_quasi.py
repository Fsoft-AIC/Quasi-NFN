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



class GLQuasiAlpha_Pooled(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default",scale = 0.1):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)
        self.out_channels = out_channels
        for i in range(L):
            if i != L-1:
                # Use aggregated statistics instead of full flattening
                # Features: mean, std, min, max of weights and biases
                
                input_dim = 14  # 7 stats each for weights and biases
                # output_dim = all_weight_shape[i][2]  # assuming this is the output channels
                self.add_module(f"layer_{i}_map_dim", nn.Linear(in_channels, out_channels))
                self.add_module(f"layer_{i}_MLP", 
                    SimpleMLP(input_dim=input_dim, output_dim=all_weight_shape[i][2], hidden_dims=[32],scale = 0.1))
    
    def forward(self, wsfeat: WeightSpaceFeatures):
        # wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        all_scale = []
        L = len(self.network_spec)
        
        for i in range(L):
            weight, bias = wsfeat[i]
            if i != L-1:
                # print("Weight: ",weight.shape)
                # print("Bias: ",bias.shape)
                # Flatten weights and biases completely for statistics
                w_flat = weight.flatten(start_dim=2)  # [batch, channels, all_weight_elements]
                b_flat = bias.flatten(start_dim=2)    # [batch, channels, all_bias_elements]
                # print("W flat: ",w_flat.shape)
                
                w_flat = getattr(self, f"layer_{i}_map_dim")(w_flat.permute(0,2,1)).permute(0,2,1)  # [batch, channels, out_channels]
                b_flat = getattr(self, f"layer_{i}_map_dim")(b_flat.permute(0,2,1)).permute(0,2,1)  # [batch, channels, out_channels]
                # Compute statistics for weights
                w_mean = w_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
                w_var = w_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]
                w_quantiles = torch.quantile(w_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=w_flat.device), dim=-1).permute(1, 2, 0)  # [batch, channels, 5]
                
                # Compute statistics for biases
                b_mean = b_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
                b_var = b_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]
                b_quantiles = torch.quantile(b_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=b_flat.device), dim=-1).permute(1, 2, 0)  # -> [5, batch, channels] -> [batch, channels, 5]
                
                # Combine all statistical features
                w_features = torch.cat([w_mean, w_var, w_quantiles], dim=-1)  # [batch, channels, 7]
                b_features = torch.cat([b_mean, b_var, b_quantiles], dim=-1).expand(w_features.shape)  # [batch, channels, 7]
                combined_stats = torch.cat([w_features, b_features], dim=-1)
                combined_stats = F.layer_norm(combined_stats, combined_stats.shape[-1:])                # print("Combined: ",combined_stats.shape)
                this_scale = getattr(self, f"layer_{i}_MLP")(combined_stats)# [batch, in_dim, out_dim] -> [batch, out_dim, in_dim]
                # print("Output proj",this_scale.shape)
                all_scale.append(this_scale)                
        return all_scale

class GLQuasiAlpha_Pooled_Output(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)
        self.out_channels = out_channels
        for i in range(L):
            if i != L-1:
                # Use aggregated statistics instead of full flattening
                # Features: mean, std, min, max of weights and biases
                
                input_dim = 14  # 7 stats each for weights and biases
                # output_dim = all_weight_shape[i][2]  # assuming this is the output channels
                self.add_module(f"layer_{i}_MLP", 
                    SimpleMLP(input_dim=input_dim, output_dim=all_weight_shape[i][2], hidden_dims=[16]))
    
    def forward(self, wsfeat: WeightSpaceFeatures):
        # wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        all_scale = []
        L = len(self.network_spec)
        
        for i in range(L):
            weight, bias = wsfeat[i]
            if i != L-1:
                # print("Weight: ",weight.shape)
                # print("Bias: ",bias.shape)
                # Flatten weights and biases completely for statistics
                w_flat = weight.flatten(start_dim=2)  # [batch, out_channels, all_weight_elements]
                b_flat = bias.flatten(start_dim=2)    # [batch, out_channels, all_bias_elements]
                # print("W flat: ",w_flat.shape)
                w_mean = w_flat.mean(dim=-1, keepdim=True)  # [batch, out_channels, 1]
                w_var = w_flat.var(dim=-1, keepdim=True)    # [batch, out_channels, 1]
                w_quantiles = torch.quantile(w_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=w_flat.device), dim=-1).permute(1, 2, 0)  # [batch, out_channels, 5]
                
                # Compute statistics for biases
                b_mean = b_flat.mean(dim=-1, keepdim=True)  # [batch, out_channels, 1]
                b_var = b_flat.var(dim=-1, keepdim=True)    # [batch, out_channels, 1]
                b_quantiles = torch.quantile(b_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=b_flat.device), dim=-1).permute(1, 2, 0)  # -> [5, batch, channels] -> [batch, out_channels, 5]
                
                # Combine all statistical features
                w_features = torch.cat([w_mean, w_var, w_quantiles], dim=-1)  # [batch, out_channels, 7]
                b_features = torch.cat([b_mean, b_var, b_quantiles], dim=-1).expand(w_features.shape)  # [batch, out_channels, 7]
                combined_stats = torch.cat([w_features, b_features], dim=-1)
                combined_stats = F.layer_norm(combined_stats, combined_stats.shape[-1:])                # print("Combined: ",combined_stats.shape)
                this_scale = getattr(self, f"layer_{i}_MLP")(combined_stats)# [batch, out_channels, out_dim]
                # print("Output proj",this_scale.shape)
                all_scale.append(this_scale)                
        return all_scale

class GLQuasiAlpha_Pooled_All(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale = 0.1):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)
        self.out_channels = out_channels
        for i in range(L):
            if i != L-1:
                # Use aggregated statistics instead of full flattening
                # Features: mean, std, min, max of weights and biases
                
                input_dim = 14  # 7 stats each for weights and biases
                # output_dim = all_weight_shape[i][2]  # assuming this is the output channels
                self.add_module(f"layer_{i}_MLP", 
                    SimpleMLP(input_dim=input_dim, output_dim=all_weight_shape[i][2], hidden_dims=[32],scale = scale))
    
    def forward(self, wsfeat: WeightSpaceFeatures):
        # wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        all_scale = []
        L = len(self.network_spec)
        
        for i in range(L):
            weight, bias = wsfeat[i]
            if i != L-1:
                # print("Weight: ",weight.shape)
                # print("Bias: ",bias.shape)
                # Flatten weights and biases completely for statistics
                w_flat = weight.flatten(start_dim=1)  # [batch,  all_weight_elements]
                b_flat = bias.flatten(start_dim=1)    # [batch, all_bias_elements]
                # print("W flat: ",w_flat.shape)
                # Compute statistics for weights
                w_mean = w_flat.mean(dim=-1, keepdim=True)  # [batch, 1]
                w_var = w_flat.var(dim=-1, keepdim=True)    # [batch, 1]
                w_quantiles = torch.quantile(w_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=w_flat.device), dim=-1).T  # [batch, 5]
                
                # Compute statistics for biases
                b_mean = b_flat.mean(dim=-1, keepdim=True)  # [batch, 1]
                b_var = b_flat.var(dim=-1, keepdim=True)    # [batch, 1]
                b_quantiles = torch.quantile(b_flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], 
                                                                device=b_flat.device), dim=-1).T  # -> [5, batch, channels] -> [batch, 5]
                
                # Combine all statistical features
                w_features = torch.cat([w_mean, w_var, w_quantiles], dim=-1)  # [batch, 7]
                b_features = torch.cat([b_mean, b_var, b_quantiles], dim=-1).expand(w_features.shape)  # [batch, 7]
                combined_stats = torch.cat([w_features, b_features], dim=-1)
                combined_stats = F.layer_norm(combined_stats, combined_stats.shape[-1:])# [batch, 14]
                # print("Combined: ",combined_stats.shape)
                this_scale = getattr(self, f"layer_{i}_MLP")(combined_stats)# [batch, out_dim] 
                # print("Output proj",this_scale.shape)
                all_scale.append(this_scale)                
        return all_scale


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256] ,activation=nn.GELU(), init_type="xavier", scale = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # not on the last layer
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation)        
        self.net = nn.Sequential(*layers)
        self.init_type = init_type
        self._initialize_weights()
        self.scale = scale 
        # self.small_factor = nn.Parameter(torch.tensor(0.01), requires_grad=False)
    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if self.init_type == "pytorch_default":
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)
                elif self.init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown init_type: {self.init_type}")

    def forward(self, x):
        raw = self.net(x)  # shape [batch_size, output_dim]

        # return F.sigmoid(raw) * self.small_factor + 1 
        return torch.sin(raw) * self.scale + 1 
        #return torch.exp(torch.clamp(self.net(x), min=-5, max=5))

def median(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    # torch.median returns values and indices
    return torch.median(x, dim=dim, keepdim=keepdim).values


def mad(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    m = median(x, dim=dim, keepdim=True)
    return median(torch.abs(x - m), dim=dim, keepdim=keepdim)


# Stable positive transforms
class PositiveTransform:
    @staticmethod
    def softplus_clamp(x: torch.Tensor, min_val=1e-3, max_val=50.0):
        return torch.clamp(F.softplus(x), min=min_val, max=max_val)

    @staticmethod
    def exp_tanh(x: torch.Tensor, min_val=1e-3, max_val=50.0):
        # bounded exponential: exp(tanh(x)) in (e^{-1}, e^{1}) approx (0.367, 2.718)
        return torch.clamp(torch.exp(torch.tanh(x)), min=min_val, max=max_val)

    @staticmethod
    def log1p_exp(x: torch.Tensor, min_val=1e-3, max_val=50.0):
        # alternative smooth transform
        return torch.clamp(torch.log1p(torch.exp(x)), min=min_val, max=max_val)


# ---------- Residual / Gated MLP ----------
class ResidualMLP(nn.Module):
    """A small, stable gated residual MLP that outputs positive values.
    Designed to be robust and easier to tune than deep vanilla MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        activation=nn.GELU(),
        init_type: str = "xavier",
        positive_transform: str = "sin",
        scale = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.positive_transform = positive_transform
        # self.small_factor = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self._init_weights(init_type)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
    def _init_weights(self, init_type: str):
        for m in [self.fc1, self.gate, self.fc_out]:
            if isinstance(m, nn.Linear):
                if init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        h = self.activation(self.norm1(self.fc1(x)))
        gated = h * torch.sigmoid(self.gate(h))  # gating for stability
        raw = self.fc_out(gated)

        return torch.sin(raw)  * self.scale + 1 


# ---------- Improved GLQuasiAlpha_Pooled ----------
class GLQuasiAlpha_Pooled_Improved(nn.Module):
    """Replacement for GLQuasiAlpha_Pooled with better stability and performance.

    Expected wsfeat input: list/tuple of (weight, bias) for each layer where
      - weight: [batch, out_channels, ...] (could be conv kernel flattened afterwards)
      - bias:   [batch, out_channels, ...]

    network_spec must provide get_all_layer_shape() -> (all_weight_shape, all_bias_shape)
      and be iterable (len(network_spec) == L)
    """

    def __init__(
        self,
        network_spec,
        in_channels: int,
        out_channels: int,
        init_type: str = "xavier",
        hidden_dim: int = 64,
        positive_transform: str = "sin",
        use_global_channel_stats: bool = False,
        scale = 0.1
    ):
        super().__init__()
        self.network_spec = network_spec
        self.c_in = in_channels
        self.c_out = out_channels
        self.init_type = init_type
        self.positive_transform = positive_transform
        self.use_global_channel_stats = use_global_channel_stats

        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)

        # We will compute for each non-final layer a small mapping MLP
        self.layer_mlps = nn.ModuleList()
        self.map_dims = nn.ModuleList()  # to map flattened ele count -> hidden projection

        # feature length: median, mad, quantiles(5) => 7 per weight/bias => 14
        base_input_dim = 14
        if use_global_channel_stats:
            # add 2 global stats (channel mean, channel mad) -> 2 values
            base_input_dim += 2

        for i in range(L):
            if i == L - 1:
                self.layer_mlps.append(None)
                self.map_dims.append(None)
                continue

            out_dim = all_weight_shape[i][2] # if len(all_weight_shape[i]) >= 3 else all_weight_shape[i][-1]
            # small projection to reduce number of elements and make quantiles robust
            # We'll map per-channel flattened elements to a lower hidden dim before stats
            self.map_dims.append(nn.Linear(in_channels, out_channels))

            # ResidualMLP will produce per-channel positive scale -> output dim = out_dim
            mlp = ResidualMLP(
                input_dim=base_input_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                init_type=init_type,
                positive_transform=positive_transform,
                scale = scale 
            )
            self.layer_mlps.append(mlp)

    def forward(self, wsfeat: List[Tuple[torch.Tensor, torch.Tensor]]):
        # wsfeat: list of (weight, bias) per layer
        all_scale = []
        L = len(self.network_spec)

        for i in range(L):
            weight, bias = wsfeat[i]
            if i == L - 1:
                break

            # Assume weight,bias shapes: [batch, channels, ...]
            # Flatten remaining dims to compute per-channel statistics
            w_flat = weight.flatten(start_dim=2)  # [B, C, Nw]
            b_flat = bias.flatten(start_dim=2)    # [B, C, Nb]

            # Project the elements dimension into channel dimension via a small linear mapping
            # We permute so linear acts on the per-element vector -> map to an ephemeral dimension
            # Using the same map_dims for both weight and bias but applied over the 'in_channels' axis
            # Here we choose a stable mapping: mean across elements + small learned linear mixing
            w_flat = self.map_dims[i](w_flat.permute(0,2,1)).permute(0,2,1)
            b_flat = self.map_dims[i](b_flat.permute(0,2,1)).permute(0,2,1)

            B, C, Nw = w_flat.shape
            _, _, Nb = b_flat.shape

            # compute robust stats for weights
            # w_med = median(w_flat, dim=-1, keepdim=True)  # [B,C,1]
            # w_mad = mad(w_flat, dim=-1, keepdim=True)     # [B,C,1]

            w_mean = w_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
            w_var = w_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]

            # quantiles: 0, .25, .5, .75, 1.0
            q_pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w_flat.device)
            w_q = torch.quantile(w_flat, q_pts, dim=-1).permute(1, 2, 0)  # -> [B,C,5]

            # b_med = median(b_flat, dim=-1, keepdim=True)
            # b_mad = mad(b_flat, dim=-1, keepdim=True)


            b_mean = b_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
            b_var = b_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]

            b_q = torch.quantile(b_flat, q_pts, dim=-1).permute(1, 2, 0)  # -> [B,C,5]

            # Assemble per-channel features (B, C, 7) each for w and b
            # w_feats = torch.cat([w_med, w_mad, w_q], dim=-1)  # [B,C,7]
            # b_feats = torch.cat([b_med, b_mad, b_q], dim=-1)  # [B,C,7]
            w_feats = torch.cat([w_mean, w_var, w_q], dim=-1)  # [B,C,7]
            b_feats = torch.cat([b_mean, b_var, b_q], dim=-1)  # [B,C,7]

            # Optionally add global channel-level summaries (mean over channels)
            if self.use_global_channel_stats:
                # channel means across channels (global scalar per sample)
                global_w_mean = w_flat.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # [B,1,1]
                global_w_mad = mad(w_flat.mean(dim=1, keepdim=True), dim=-1, keepdim=True)    # [B,1,1]
                global_feats = torch.cat([global_w_mean, global_w_mad], dim=-1)  # [B,1,2]
                # Broadcast to per-channel
                global_feats_exp = global_feats.expand(-1, C, -1)  # [B,C,2]
                combined = torch.cat([w_feats, b_feats, global_feats_exp], dim=-1)  # [B,C,16]
            else:
                combined = torch.cat([w_feats, b_feats], dim=-1)  # [B,C,14]

            # Normalize features per-channel for stability
            combined = F.layer_norm(combined, combined.shape[-1:])

            # Feed through MLP per-channel: flatten batch+channels into one leading dim
            Bc = B * C
            combined_flat = combined.reshape(Bc, -1)  # [B*C, D]

            mlp = self.layer_mlps[i]
            this_scale = mlp(combined_flat)  # [B*C, out_dim]

            # Reshape back: [B, C, out_dim]
            out_dim = this_scale.shape[-1]
            this_scale = this_scale.view(B, C, out_dim)

            all_scale.append(this_scale)

        return all_scale

class GLQuasiAlpha_Pooled_Combined_Per(nn.Module):
    """Replacement for GLQuasiAlpha_Pooled with better stability and performance.

    Expected wsfeat input: list/tuple of (weight, bias) for each layer where
      - weight: [batch, out_channels, ...] (could be conv kernel flattened afterwards)
      - bias:   [batch, out_channels, ...]

    network_spec must provide get_all_layer_shape() -> (all_weight_shape, all_bias_shape)
      and be iterable (len(network_spec) == L)
    """

    def __init__(
        self,
        network_spec,
        in_channels: int,
        out_channels: int,
        init_type: str = "xavier",
        hidden_dim: int = 64,
        positive_transform: str = "sin",
        use_global_channel_stats: bool = False,
        scale = 0.1
    ):
        super().__init__()
        self.network_spec = network_spec
        self.c_in = in_channels
        self.c_out = out_channels
        self.init_type = init_type
        self.positive_transform = positive_transform
        self.use_global_channel_stats = use_global_channel_stats

        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)

        # We will compute for each non-final layer a small mapping MLP
        self.layer_mlps = nn.ModuleList()
        self.map_dims = nn.ModuleList()  # to map flattened ele count -> hidden projection

        # feature length: median, mad, quantiles(5) => 7 per weight/bias => 14
        base_input_dim = 14
        if use_global_channel_stats:
            # add 2 global stats (channel mean, channel mad) -> 2 values
            base_input_dim += 2
        base_input_dim *= L-1
        for i in range(L):
            if i == L - 1:
                self.layer_mlps.append(None)
                self.map_dims.append(None)
                continue

            out_dim = all_weight_shape[i][2] # if len(all_weight_shape[i]) >= 3 else all_weight_shape[i][-1]
            # small projection to reduce number of elements and make quantiles robust
            # We'll map per-channel flattened elements to a lower hidden dim before stats
            self.map_dims.append(nn.Linear(in_channels, out_channels))

            # ResidualMLP will produce per-channel positive scale -> output dim = out_dim
            mlp = ResidualMLP(
                input_dim=base_input_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                init_type=init_type,
                positive_transform=positive_transform,
                scale = scale 
            )
            self.layer_mlps.append(mlp)

    def forward(self, wsfeat: List[Tuple[torch.Tensor, torch.Tensor]]):
        # wsfeat: list of (weight, bias) per layer
        all_scale = []
        L = len(self.network_spec)

        all_combined = []
        for i in range(L):
            weight, bias = wsfeat[i]
            if i == L - 1:
                break

            # Assume weight,bias shapes: [batch, channels, ...]
            # Flatten remaining dims to compute per-channel statistics
            w_flat = weight.flatten(start_dim=2)  # [B, C, Nw]
            b_flat = bias.flatten(start_dim=2)    # [B, C, Nb]

            # Project the elements dimension into channel dimension via a small linear mapping
            # We permute so linear acts on the per-element vector -> map to an ephemeral dimension
            # Using the same map_dims for both weight and bias but applied over the 'in_channels' axis
            # Here we choose a stable mapping: mean across elements + small learned linear mixing
            w_flat = self.map_dims[i](w_flat.permute(0,2,1)).permute(0,2,1)
            b_flat = self.map_dims[i](b_flat.permute(0,2,1)).permute(0,2,1)

            B, C, Nw = w_flat.shape
            _, _, Nb = b_flat.shape

            # compute robust stats for weights
            # w_med = median(w_flat, dim=-1, keepdim=True)  # [B,C,1]
            # w_mad = mad(w_flat, dim=-1, keepdim=True)     # [B,C,1]

            w_mean = w_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
            w_var = w_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]

            # quantiles: 0, .25, .5, .75, 1.0
            q_pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w_flat.device)
            w_q = torch.quantile(w_flat, q_pts, dim=-1).permute(1, 2, 0)  # -> [B,C,5]

            # b_med = median(b_flat, dim=-1, keepdim=True)
            # b_mad = mad(b_flat, dim=-1, keepdim=True)


            b_mean = b_flat.mean(dim=-1, keepdim=True)  # [batch, channels, 1]
            b_var = b_flat.var(dim=-1, keepdim=True)    # [batch, channels, 1]

            b_q = torch.quantile(b_flat, q_pts, dim=-1).permute(1, 2, 0)  # -> [B,C,5]

            # Assemble per-channel features (B, C, 7) each for w and b
            # w_feats = torch.cat([w_med, w_mad, w_q], dim=-1)  # [B,C,7]
            # b_feats = torch.cat([b_med, b_mad, b_q], dim=-1)  # [B,C,7]
            w_feats = torch.cat([w_mean, w_var, w_q], dim=-1)  # [B,C,7]
            b_feats = torch.cat([b_mean, b_var, b_q], dim=-1)  # [B,C,7]

            # Optionally add global channel-level summaries (mean over channels)
            if self.use_global_channel_stats:
                # channel means across channels (global scalar per sample)
                global_w_mean = w_flat.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # [B,1,1]
                global_w_mad = mad(w_flat.mean(dim=1, keepdim=True), dim=-1, keepdim=True)    # [B,1,1]
                global_feats = torch.cat([global_w_mean, global_w_mad], dim=-1)  # [B,1,2]
                # Broadcast to per-channel
                global_feats_exp = global_feats.expand(-1, C, -1)  # [B,C,2]
                combined = torch.cat([w_feats, b_feats, global_feats_exp], dim=-1)  # [B,C,16]
            else:
                combined = torch.cat([w_feats, b_feats], dim=-1)  # [B,C,14]

            # Normalize features per-channel for stability
            combined = F.layer_norm(combined, combined.shape[-1:])

            # Feed through MLP per-channel: flatten batch+channels into one leading dim
            Bc = B * C
            combined_flat = combined.reshape(Bc, -1)  # [B*C, D]
            all_combined.append(combined_flat)

        all_combined = torch.cat(all_combined, dim=-1)
        for i in range(L):
            if i == L - 1:
                break

            mlp = self.layer_mlps[i]
            this_scale = mlp(all_combined)  # [B*C, out_dim]

            # Reshape back: [B, C, out_dim]
            out_dim = this_scale.shape[-1]
            this_scale = this_scale.view(B, C, out_dim)

            all_scale.append(this_scale)

        return all_scale


class GLQuasiAlpha_Pooled_All_Improved(nn.Module):
    """Replacement for GLQuasiAlpha_Pooled with better stability and performance.

    Expected wsfeat input: list/tuple of (weight, bias) for each layer where
      - weight: [batch, out_channels, ...] (could be conv kernel flattened afterwards)
      - bias:   [batch, out_channels, ...]

    network_spec must provide get_all_layer_shape() -> (all_weight_shape, all_bias_shape)
      and be iterable (len(network_spec) == L)
    """

    def __init__(
        self,
        network_spec,
        in_channels: int,
        out_channels: int,
        init_type: str = "xavier",
        hidden_dim: int = 64,
        positive_transform: str = "sin",
        use_global_channel_stats: bool = True,
        scale = 0.1, 
    ):
        super().__init__()
        self.network_spec = network_spec
        self.c_in = in_channels
        self.c_out = out_channels
        self.init_type = init_type
        self.positive_transform = positive_transform
        self.use_global_channel_stats = use_global_channel_stats

        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)

        # We will compute for each non-final layer a small mapping MLP
        self.layer_mlps = nn.ModuleList()
        self.map_dims = nn.ModuleList()  # to map flattened ele count -> hidden projection

        # feature length: median, mad, quantiles(5) => 7 per weight/bias => 14
        base_input_dim = 14
        if use_global_channel_stats:
            # add 2 global stats (channel mean, channel mad) -> 2 values
            base_input_dim += 2

        for i in range(L):
            if i == L - 1:
                self.layer_mlps.append(None)
                self.map_dims.append(None)
                continue

            out_dim = all_weight_shape[i][2] # if len(all_weight_shape[i]) >= 3 else all_weight_shape[i][-1]
            # small projection to reduce number of elements and make quantiles robust
            # We'll map per-channel flattened elements to a lower hidden dim before stats
            self.map_dims.append(nn.Linear(in_channels, out_channels))

            # ResidualMLP will produce per-channel positive scale -> output dim = out_dim
            mlp = ResidualMLP(
                input_dim=base_input_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                init_type=init_type,
                positive_transform=positive_transform,
                scale=scale
            )
            self.layer_mlps.append(mlp)

    def forward(self, wsfeat: List[Tuple[torch.Tensor, torch.Tensor]]):
        # wsfeat: list of (weight, bias) per layer
        all_scale = []
        L = len(self.network_spec)

        for i in range(L):
            weight, bias = wsfeat[i]
            if i == L - 1:
                break

            # Assume weight,bias shapes: [batch, channels, ...]
            # Flatten remaining dims to compute per-channel statistics
            w_flat = weight.flatten(start_dim=1)  # [B, all_weight_elements]
            b_flat = bias.flatten(start_dim=1)    # [B, all_weight_elements]

            # Project the elements dimension into channel dimension via a small linear mapping
            # We permute so linear acts on the per-element vector -> map to an ephemeral dimension
            # Using the same map_dims for both weight and bias but applied over the 'in_channels' axis
            # Here we choose a stable mapping: mean across elements + small learned linear mixing
            B, Nw = w_flat.shape
            _, Nb = b_flat.shape

            # compute robust stats for weights
            w_med = median(w_flat, dim=-1, keepdim=True)  # [B,1]
            w_mad = mad(w_flat, dim=-1, keepdim=True)     # [B,1]
            # quantiles: 0, .25, .5, .75, 1.0
            q_pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w_flat.device)
            w_q = torch.quantile(w_flat, q_pts, dim=-1).T  # -> [B,5]

            b_med = median(b_flat, dim=-1, keepdim=True)
            b_mad = mad(b_flat, dim=-1, keepdim=True)
            b_q = torch.quantile(b_flat, q_pts, dim=-1).T  # -> [B,5]

            # Assemble per-channel features (B, C, 7) each for w and b
            w_feats = torch.cat([w_med, w_mad, w_q], dim=-1)  # [B,7]
            b_feats = torch.cat([b_med, b_mad, b_q], dim=-1)  # [B,7]

            # Optionally add global channel-level summaries (mean over channels)
            if self.use_global_channel_stats:
                # channel means across channels (global scalar per sample)
                global_w_mean = w_flat.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # [B,1]
                global_w_mad = mad(w_flat.mean(dim=1, keepdim=True), dim=-1, keepdim=True)    # [B,1]
                global_feats = torch.cat([global_w_mean, global_w_mad], dim=-1)  # [B,2]
                # Broadcast to per-channel
                combined = torch.cat([w_feats, b_feats, global_feats], dim=-1)  # [B,16]
            else:
                combined = torch.cat([w_feats, b_feats], dim=-1)  # [B,14]

            # Normalize features per-channel for stability
            combined = F.layer_norm(combined, combined.shape[-1:])

            # Feed through MLP per-channel: flatten batch+channels into one leading dim
            mlp = self.layer_mlps[i]
            this_scale = mlp(combined)  # [B, out_dim]


            all_scale.append(this_scale)

        return all_scale


class GLQuasiAlpha_Pooled_Combined_All(nn.Module):
    """Replacement for GLQuasiAlpha_Pooled with better stability and performance.

    Expected wsfeat input: list/tuple of (weight, bias) for each layer where
      - weight: [batch, out_channels, ...] (could be conv kernel flattened afterwards)
      - bias:   [batch, out_channels, ...]

    network_spec must provide get_all_layer_shape() -> (all_weight_shape, all_bias_shape)
      and be iterable (len(network_spec) == L)
    """

    def __init__(
        self,
        network_spec,
        in_channels: int,
        out_channels: int,
        init_type: str = "xavier",
        hidden_dim: int = 64,
        positive_transform: str = "sin",
        use_global_channel_stats: bool = True,
        scale = 0.1, 
    ):
        super().__init__()
        self.network_spec = network_spec
        self.c_in = in_channels
        self.c_out = out_channels
        self.init_type = init_type
        self.positive_transform = positive_transform
        self.use_global_channel_stats = use_global_channel_stats

        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)

        # We will compute for each non-final layer a small mapping MLP
        self.layer_mlps = nn.ModuleList()

        # feature length: median, mad, quantiles(5) => 7 per weight/bias => 14
        base_input_dim = 14
        if use_global_channel_stats:
            # add 2 global stats (channel mean, channel mad) -> 2 values
            base_input_dim += 2
        base_input_dim *= L-1
        for i in range(L):
            if i == L - 1:
                self.layer_mlps.append(None)
                continue

            out_dim = all_weight_shape[i][2] # if len(all_weight_shape[i]) >= 3 else all_weight_shape[i][-1]
            # small projection to reduce number of elements and make quantiles robust
            # We'll map per-channel flattened elements to a lower hidden dim before stats

            # ResidualMLP will produce per-channel positive scale -> output dim = out_dim
            mlp = ResidualMLP(
                input_dim=base_input_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                init_type=init_type,
                positive_transform=positive_transform,
                scale=scale
            )
            self.layer_mlps.append(mlp)

    def forward(self, wsfeat: List[Tuple[torch.Tensor, torch.Tensor]]):
        # wsfeat: list of (weight, bias) per layer
        all_scale = []
        L = len(self.network_spec)

        all_combined = []
        for i in range(L):
            weight, bias = wsfeat[i]
            if i == L - 1:
                break

            # Assume weight,bias shapes: [batch, channels, ...]
            # Flatten remaining dims to compute per-channel statistics
            w_flat = weight.flatten(start_dim=1)  # [B, all_weight_elements]
            b_flat = bias.flatten(start_dim=1)    # [B, all_weight_elements]

            # Project the elements dimension into channel dimension via a small linear mapping
            # We permute so linear acts on the per-element vector -> map to an ephemeral dimension
            # Using the same map_dims for both weight and bias but applied over the 'in_channels' axis
            # Here we choose a stable mapping: mean across elements + small learned linear mixing
            B, Nw = w_flat.shape
            _, Nb = b_flat.shape

            # compute robust stats for weights
            w_med = median(w_flat, dim=-1, keepdim=True)  # [B,1]
            w_mad = mad(w_flat, dim=-1, keepdim=True)     # [B,1]
            # quantiles: 0, .25, .5, .75, 1.0
            q_pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w_flat.device)
            w_q = torch.quantile(w_flat, q_pts, dim=-1).T  # -> [B,5]

            b_med = median(b_flat, dim=-1, keepdim=True)
            b_mad = mad(b_flat, dim=-1, keepdim=True)
            b_q = torch.quantile(b_flat, q_pts, dim=-1).T  # -> [B,5]

            # Assemble per-channel features (B, C, 7) each for w and b
            w_feats = torch.cat([w_med, w_mad, w_q], dim=-1)  # [B,7]
            b_feats = torch.cat([b_med, b_mad, b_q], dim=-1)  # [B,7]

            # Optionally add global channel-level summaries (mean over channels)
            if self.use_global_channel_stats:
                # channel means across channels (global scalar per sample)
                global_w_mean = w_flat.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # [B,1]
                global_w_mad = mad(w_flat.mean(dim=1, keepdim=True), dim=-1, keepdim=True)    # [B,1]
                global_feats = torch.cat([global_w_mean, global_w_mad], dim=-1)  # [B,2]
                # Broadcast to per-channel
                combined = torch.cat([w_feats, b_feats, global_feats], dim=-1)  # [B,16]
            else:
                combined = torch.cat([w_feats, b_feats], dim=-1)  # [B,14]

            # Normalize features per-channel for stability
            combined = F.layer_norm(combined, combined.shape[-1:])
            all_combined.append(combined)

        all_combined = torch.cat(all_combined, dim = -1)
        for i in range(L):
            if i == L - 1:
                break

            # Feed through MLP per-channel: flatten batch+channels into one leading dim
            mlp = self.layer_mlps[i]
            this_scale = mlp(all_combined)  # [B, out_dim]


            all_scale.append(this_scale)

        return all_scale
def compute_spectral_features(x: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """Compute spectral features via FFT (for 1D signals along last dim)."""
    if x.shape[-1] < 4:
        # Too small for meaningful FFT
        return torch.zeros(*x.shape[:-1], top_k * 2, device=x.device, dtype=x.dtype)
    
    # Apply FFT along last dimension
    fft = torch.fft.rfft(x, dim=-1)
    magnitudes = torch.abs(fft)
    phases = torch.angle(fft)
    
    # Get top-k frequencies by magnitude
    top_mags, indices = torch.topk(magnitudes, min(top_k, magnitudes.shape[-1]), dim=-1)
    top_phases = torch.gather(phases, -1, indices)
    
    # Concatenate magnitude and phase features
    return torch.cat([top_mags, top_phases], dim=-1)

def compute_correlation_features(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute correlation-based features between weights and biases."""
    # Ensure compatible shapes by broadcasting or pooling
    w_mean = w.mean(dim=-1, keepdim=True)
    b_mean = b.mean(dim=-1, keepdim=True)
    
    # Simple correlation proxy: normalized dot product
    w_norm = F.normalize(w - w_mean, p=2, dim=-1)
    b_norm = F.normalize(b - b_mean, p=2, dim=-1)
    
    # If shapes don't match, pool to smaller size
    if w_norm.shape[-1] != b_norm.shape[-1]:
        target_size = min(w_norm.shape[-1], b_norm.shape[-1])
        if w_norm.shape[-1] > target_size:
            w_norm = F.adaptive_avg_pool1d(w_norm.unsqueeze(1), target_size).squeeze(1)
        if b_norm.shape[-1] > target_size:
            b_norm = F.adaptive_avg_pool1d(b_norm.unsqueeze(1), target_size).squeeze(1)
    
    correlation = (w_norm * b_norm).sum(dim=-1, keepdim=True)
    return correlation

def compute_distribution_features(x: torch.Tensor) -> torch.Tensor:
    """Compute higher-order distribution features."""
    eps = 1e-8
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + eps
    
    # Standardized moments
    x_std = (x - mean) / std
    
    # Skewness (3rd moment)
    skewness = (x_std ** 3).mean(dim=-1, keepdim=True)
    
    # Kurtosis (4th moment)
    kurtosis = (x_std ** 4).mean(dim=-1, keepdim=True)
    
    # Inter-quartile range
    q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
    q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
    iqr = q75 - q25
    
    # Coefficient of variation
    cv = std / (torch.abs(mean) + eps)
    
    return torch.cat([skewness, kurtosis, iqr, cv], dim=-1)

# ---------- Enhanced GLQuasiAlpha_Pooled ----------
class GLQuasiAlpha_Pooled_Enhanced(nn.Module):
    """Enhanced version with richer feature extraction and better noise modeling."""
    
    def __init__(
        self,
        network_spec,
        in_channels: int,
        out_channels: int,
        init_type: str = "xavier",
        hidden_dim: int = 64,
        positive_transform: str = "softplus_clamp",
        use_global_channel_stats: bool = True,
        use_spectral_features: bool = True,
        use_correlation_features: bool = True,
        use_distribution_features: bool = True,
        scale = 0.1
    ):
        super().__init__()
        self.network_spec = network_spec
        self.c_in = in_channels
        self.c_out = out_channels
        self.init_type = init_type
        self.positive_transform = positive_transform
        self.use_global_channel_stats = use_global_channel_stats
        self.use_spectral_features = use_spectral_features
        self.use_correlation_features = use_correlation_features
        self.use_distribution_features = use_distribution_features
        
        all_weight_shape, all_bias_shape = network_spec.get_all_layer_shape()
        L = len(network_spec)
        
        # Calculate feature dimension
        base_input_dim = 14  # Original: median, mad, quantiles(5) for both w and b
        
        if use_global_channel_stats:
            base_input_dim += 4  # Enhanced: global mean, mad, min, max
        
        if use_spectral_features:
            base_input_dim += 12  # 3 top frequencies * 2 (mag+phase) * 2 (w+b)
        
        if use_correlation_features:
            base_input_dim += 1  # Weight-bias correlation
        
        if use_distribution_features:
            base_input_dim += 8  # 4 features each for w and b
        
        self.layer_mlps = nn.ModuleList()
        self.map_dims = nn.ModuleList()
        
        
        for i in range(L):
            if i == L - 1:
                self.layer_mlps.append(None)
                self.map_dims.append(None)
                continue
            
            out_dim = all_weight_shape[i][2]
            
            # Channel projection
            self.map_dims.append(nn.Linear(in_channels, out_channels))
            
            # Enhanced MLP with sinusoidal encoding
            mlp = ResidualMLP(
                input_dim=base_input_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                init_type=init_type,
                positive_transform=positive_transform,
                scale = scale
            )
            self.layer_mlps.append(mlp)
            
    
    def forward(self, wsfeat: List[Tuple[torch.Tensor, torch.Tensor]]):
        all_scale = []
        L = len(self.network_spec)
        
        for i in range(L):
            weight, bias = wsfeat[i]
            if i == L - 1:
                break
            
            # Flatten spatial dimensions
            w_flat = weight.flatten(start_dim=2)  # [B, C, Nw]
            b_flat = bias.flatten(start_dim=2)    # [B, C, Nb]
            
            # Project elements
            w_flat = self.map_dims[i](w_flat.permute(0, 2, 1)).permute(0, 2, 1)
            b_flat = self.map_dims[i](b_flat.permute(0, 2, 1)).permute(0, 2, 1)
            
            B, C, Nw = w_flat.shape
            
            # ========== Extract Enhanced Features ==========
            features = []
            
            # 1. Basic statistics (original features)
            w_med = median(w_flat, dim=-1, keepdim=True)
            w_mad = mad(w_flat, dim=-1, keepdim=True)
            q_pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w_flat.device)
            w_q = torch.quantile(w_flat, q_pts, dim=-1).permute(1, 2, 0)
            
            b_med = median(b_flat, dim=-1, keepdim=True)
            b_mad = mad(b_flat, dim=-1, keepdim=True)
            b_q = torch.quantile(b_flat, q_pts, dim=-1).permute(1, 2, 0)
            
            features.extend([w_med, w_mad, w_q, b_med, b_mad, b_q])
            
            # 2. Global channel statistics (enhanced)
            if self.use_global_channel_stats:
                global_w_mean = w_flat.mean(dim=(1, 2), keepdim=True).expand(-1, C, -1)
                global_w_mad = mad(w_flat.reshape(B, -1), dim=-1, keepdim=True).unsqueeze(1).expand(-1, C, -1)
                global_w_min = w_flat.min(dim=1, keepdim=True)[0].min(dim=-1, keepdim=True).expand(-1, C, -1)
                global_w_max = w_flat.max(dim=1, keepdim=True)[0].max(dim=-1, keepdim=True).expand(-1, C, -1)
                features.extend([global_w_mean, global_w_mad, global_w_min, global_w_max])
            
            # 3. Spectral features
            if self.use_spectral_features:
                w_spectral = compute_spectral_features(w_flat, top_k=3)
                b_spectral = compute_spectral_features(b_flat, top_k=3)
                features.extend([w_spectral, b_spectral])
            
            # 4. Correlation features
            if self.use_correlation_features:
                wb_correlation = compute_correlation_features(w_flat, b_flat)
                features.append(wb_correlation)
            
            # 5. Distribution features
            if self.use_distribution_features:
                w_dist = compute_distribution_features(w_flat)
                b_dist = compute_distribution_features(b_flat)
                features.extend([w_dist, b_dist])
            
            # Concatenate all features
            combined = torch.cat(features, dim=-1)  # [B, C, feature_dim]
            
            # Normalize for stability
            combined = F.layer_norm(combined, combined.shape[-1:])
            # Process through MLP
            Bc = B * C
            combined_flat = combined.reshape(Bc, -1)
            
            mlp = self.layer_mlps[i]
            this_scale = mlp(combined_flat)
            
            # Reshape back
            out_dim = this_scale.shape[-1]
            this_scale = this_scale.view(B, C, out_dim)
            
            all_scale.append(this_scale)
        
        return all_scale