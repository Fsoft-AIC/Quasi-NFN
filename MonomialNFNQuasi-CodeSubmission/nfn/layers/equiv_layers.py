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


class Pointwise(nn.Module):
    """Assumes full row/col exchangeability of weights in each layer."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels):
        super().__init__()
        self.network_spec = network_spec
        self.in_channels = in_channels
        self.out_channels = out_channels
        # register num_layers in_channels -> out_channels linear layers
        self.weight_maps, self.bias_maps = nn.ModuleList(), nn.ModuleList()
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        for i in range(len(network_spec)):
            fac_i = filter_facs[i]
            self.weight_maps.append(nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            self.bias_maps.append(nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        # weights is a list of tensors, each with shape (B, C_in, nrow, ncol)
        # each tensor is reshaped to (B, nrow * ncol, C_in), passed through a linear
        # layer, then reshaped back to (B, C_out, nrow, ncol)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out_weights.append(self.weight_maps[i](weight))
            out_biases.append(self.bias_maps[i](bias))
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"Pointwise(in_channels={self.in_channels}, out_channels={self.out_channels})"


class NPLinear(nn.Module):
    """Assume permutation symmetry of input and output layers, as well as hidden."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, io_embed=False, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = L + sum(filter_facs)
        for i in range(L):
            fac_i = filter_facs[i]
            # pointwise
            self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            # broadcasts over rows and columns
            self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

            # broadcasts over rows or columns
            row_in, col_in = fac_i * in_channels, (fac_i + 1) * in_channels
            if i > 0:
                fac_im1 = filter_facs[i - 1]
                row_in += (fac_im1 + 1) * in_channels
            if i < len(network_spec) - 1:
                fac_ip1 = filter_facs[i + 1]
                col_in += fac_ip1 * in_channels
            self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
            self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))

            # pointwise
            self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
            set_init_(
                getattr(self, f"layer_{i}"),
                getattr(self, f"layer_{i}_rc"),
                getattr(self, f"layer_{i}_r"),
                getattr(self, f"layer_{i}_c"),
                init_type=init_type,
            )
            set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"), init_type=init_type)
        self.io_embed = io_embed
        if io_embed:
            # initialize learned position embeddings to break input and output symmetry
            n_in, n_out = network_spec.get_io()
            self.in_embed = nn.Parameter(torch.randn(1, filter_facs[0] * in_channels, 1, n_in))
            self.out_embed = nn.Parameter(torch.randn(1, filter_facs[-1] * in_channels, n_out, 1))
            self.out_bias_embed = nn.Parameter(torch.randn(1, in_channels, n_out))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        if self.io_embed:
            new_weights = (wsfeat.weights[0] + self.in_embed, *wsfeat.weights[1:-1], wsfeat.weights[-1] + self.out_embed)
            new_biases = (*wsfeat.biases[:-1], wsfeat.biases[-1] + self.out_bias_embed)
            wsfeat = WeightSpaceFeatures(new_weights, new_biases)
        weights, biases = wsfeat.weights, wsfeat.biases
        # weights[i] shape: [B, C, row, collumn]
        row_means = [w.mean(dim=-2) for w in weights]
        col_means = [w.mean(dim=-1) for w in weights]
        rowcol_means = [w.mean(dim=(-2, -1)) for w in weights]  # (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # (B, C_in)
        wb_means = torch.cat(rowcol_means + bias_means, dim=-1)  # (B, 2 * C_in * n_layers)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            # weights shape: [B, C, row, collumn]
            z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
            z2 = getattr(self, f"layer_{i}_rc")(wb_means)[..., None, None]
            row_bdcst = [row_means[i]]  # (B, C_in, ncol)
            col_bdcst = [col_means[i], bias]  # (B, 2 * C_in, nrow)
            if i > 0:
                row_bdcst.extend([col_means[i-1], biases[i-1]])  # (B, C_in, ncol)
            if i < len(wsfeat.weights) - 1:
                col_bdcst.append(row_means[i+1])  # (B, C_in, nrow)
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"layer_{i}_r")(torch.cat(row_bdcst, dim=-2)).unsqueeze(-2)  # (B, C_out, 1, ncol)
            z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
            out_weights.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
            u2 = getattr(self, f"bias_{i}_rc")(wb_means).unsqueeze(-1)  # (B, C_out, 1)
            out_biases.append(u1 + u2)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"NPLinear(in_channels={self.c_in}, out_channels={self.c_out})"


class HNPLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.network_spec = network_spec
        n_in, n_out = network_spec.get_io()
        self.n_in, self.n_out = n_in, n_out
        self.c_in, self.c_out = in_channels, out_channels
        self.L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = n_in * filter_facs[0] + n_out * filter_facs[-1] + self.L + n_out
        for i in range(self.L):
            n_rc_inp += filter_facs[i]
        for i in range(self.L):
            fac_i = filter_facs[i]
            if i == 0:
                fac_ip1 = filter_facs[i+1]
                rpt_in = (fac_i * n_in + fac_ip1 + 1)
                if i + 1 == self.L - 1:
                    rpt_in += n_out * fac_ip1
                self.w0_rpt = nn.Conv1d(rpt_in * in_channels, n_in * fac_i * out_channels, 1)
                self.w0_rbdcst = nn.Linear(n_rc_inp * in_channels, n_in * fac_i * out_channels)

                self.bias_0 = nn.Conv1d(rpt_in * in_channels, out_channels, 1)
                self.bias_0_rc = nn.Linear(n_rc_inp * in_channels, out_channels)
                set_init_(self.bias_0, self.bias_0_rc)
            elif i == self.L - 1:
                fac_im1 = filter_facs[i-1]
                cpt_in = (fac_i * n_out + fac_im1)
                if i - 1 == 0:
                    cpt_in += n_in * fac_im1
                self.wfin_cpt = nn.Conv1d(cpt_in * in_channels, n_out * fac_i * out_channels, 1)
                self.wfin_cbdcst = nn.Linear(n_rc_inp * in_channels, n_out * fac_i * out_channels)
                set_init_(self.wfin_cpt, self.wfin_cbdcst)

                self.bias_fin_rc = nn.Linear(n_rc_inp * in_channels, n_out * out_channels)
            else:
                self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

                fac_im1, fac_ip1 = filter_facs[i-1], filter_facs[i+1]
                row_in, col_in = (fac_im1 + fac_i + 1) * in_channels, (fac_ip1 + fac_i + 1) * in_channels
                if i == 1: row_in += n_in * filter_facs[0] * in_channels
                if i == self.L - 2: col_in += n_out * filter_facs[-1] * in_channels
                self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))
                set_init_(
                    getattr(self, f"layer_{i}"),
                    getattr(self, f"layer_{i}_rc"),
                    getattr(self, f"layer_{i}_r"),
                    getattr(self, f"layer_{i}_c"),
                )

                self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
                self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
                set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"))
        self.rearr1_wt1 = Rearrange("b c_in nrow ncol -> b (c_in ncol) nrow")
        self.rearr1_wtL = Rearrange("b c_in n_out nrow -> b (c_in n_out) nrow")
        self.rearr1_outwt = Rearrange("b (c_out ncol) nrow -> b c_out nrow ncol", ncol=n_in)
        self.rearrL_wtL = Rearrange("b c_in nrow ncol -> b (c_in nrow) ncol")
        self.rearrL_wt1 = Rearrange("b c_in ncol n_in -> b (c_in n_in) ncol")
        self.rearrL_outwt = Rearrange("b (c_out nrow) ncol -> b c_out nrow ncol", nrow=n_out)
        self.rearrL_outbs = Rearrange("b (c_out nrow) -> b c_out nrow", nrow=n_out)
        self.rearr2_wt1 = Rearrange("b c ncol n_in -> b (c n_in) ncol")
        self.rearrLm1_wtL = Rearrange("b c n_out nrow -> b (c n_out) nrow")

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        weights, biases = wsfeat.weights, wsfeat.biases
        col_means = [w.mean(dim=-1) for w in weights]  # shapes: (B, C_in, nrow_i=ncol_i+1)
        row_means = [w.mean(dim=-2) for w in weights]  # shapes: (B, C_in, ncol_i=nrow_i-1)
        rc_means = [w.mean(dim=(-1, -2)) for w in weights]  # shapes: (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # shapes: (B, C_in)
        rm0 = torch.flatten(row_means[0], start_dim=-2)  # b c_in ncol -> b (c_in ncol)
        cmL = torch.flatten(col_means[-1], start_dim=-2)  # b c_in nrowL -> b (c_in nrowL)
        final_bias = torch.flatten(biases[-1], start_dim=-2)  # b c_in nrow -> b (c_in nrow)
        # (B, C_in * (2 * L + n_in + 2 * n_out))
        rc_inp = torch.cat(rc_means + bias_means + [rm0, cmL, final_bias], dim=-1)

        out_weights, out_biases = [], []
        for i in range(self.L):
            weight, bias = wsfeat[i]
            if i == 0:
                rpt = [self.rearr1_wt1(weight), row_means[1], bias]
                if i+1 == self.L - 1:
                    rpt.append(self.rearr1_wtL(weights[-1]))
                rpt = torch.cat(rpt, dim=-2)  # repeat ptwise across rows
                z1 = self.w0_rpt(rpt)
                z2 = self.w0_rbdcst(rc_inp)[..., None]  # (b, c_out * ncol, 1)
                z = self.rearr1_outwt(z1 + z2)
                u1 = self.bias_0(rpt)  # (B, C_out, nrow)
                u2 = self.bias_0_rc(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            elif i == self.L - 1:
                cpt = [self.rearrL_wtL(weight), col_means[-2]]  # b c_in ncol
                if i - 1 == 0:
                    cpt.append(self.rearrL_wt1(weights[0]))
                z1 = self.wfin_cpt(torch.cat(cpt, dim=-2))  # (B, C_out * nrow, ncol)
                z2 = self.wfin_cbdcst(rc_inp)[..., None]  # (b, c_out * nrow, 1)
                z = self.rearrL_outwt(z1 + z2)
                u = self.rearrL_outbs(self.bias_fin_rc(rc_inp))
            else:
                z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
                z2 = getattr(self, f"layer_{i}_rc")(rc_inp)[..., None, None]
                row_bdcst = [row_means[i], col_means[i-1], biases[i-1]]
                col_bdcst = [col_means[i], bias, row_means[i+1]]
                if i == 1:
                    row_bdcst.append(self.rearr2_wt1(weights[0]))
                if i == len(weights) - 2:
                    col_bdcst.append(self.rearrLm1_wtL(weights[-1]))
                row_bdcst = torch.cat(row_bdcst, dim=-2)  # (B, (3 + ?n_in) * C_in, ncol)
                col_bdcst = torch.cat(col_bdcst, dim=-2)  # (B, (3 + ?n_out) * C_in, nrow)
                z3 = getattr(self, f"layer_{i}_r")(row_bdcst).unsqueeze(-2)  # (B, C_out, 1, ncol)
                z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
                z = z1 + z2 + z3 + z4
                u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
                u2 = getattr(self, f"bias_{i}_rc")(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            out_weights.append(z)
            out_biases.append(u)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"HNPLinear(in_channels={self.c_in}, out_channels={self.c_out}, n_in={self.n_in}, n_out={self.n_out}, num_layers={self.L})"



def simple_attention(q, k, v, dropout=None):
    # q, k, v: (..., T, d)
    # TODO: consider replacing with F.scaled_dot_product_attention. But probably won't get
    # Flash Attention speedup unless we have A100 @ 16bit: https://github.com/pytorch/pytorch/pull/81434#issuecomment-1451120687.
    attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v


class ChannelLinear(nn.Module):
    """Probably equivalent to a 1x1 nn.Conv2d."""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.channels_last = Rearrange("b c ... -> b ... c")
        self.channels_first = Rearrange("b ... c -> b c ...")

    def forward(self, x):
        x = self.channels_last(x)
        x = self.linear(x)
        return self.channels_first(x)


class NPAttention(nn.Module):
    def __init__(
        self, network_spec: NetworkSpec, channels,
        num_heads=8, dropout=0,
        share_projections=True,
        ablate_crossterm=False,
        ablate_diagonalterm=False,
    ):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})."
        self.network_spec = network_spec
        self.share_projections = share_projections
        if share_projections:
            self.to_qkv = ChannelLinear(channels, 3 * channels)
        else:
            self.weight_to_qkv = nn.ModuleList()  # maps channel dim k * k * c --> c
            self.bias_to_qkv = nn.ModuleList()
            self.unproject_weight = nn.ModuleList()  # maps channel dim c --> k * k * c
            filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
            for filter_fac in filter_facs:
                self.weight_to_qkv.append(ChannelLinear(filter_fac * channels, 3 * channels))
                self.bias_to_qkv.append(ChannelLinear(channels, 3 * channels))
                self.unproject_weight.append(ChannelLinear(channels, filter_fac * channels))
        self.nh = num_heads
        self.ch = channels // num_heads  # channels per head
        self.dropout = nn.Dropout(dropout)
        self.split_heads = Rearrange("b (nh ch) ... -> b nh ch ...", nh=self.nh, ch=3*self.ch)
        self.combine_nh_nc = Rearrange("b nh t c -> b t (nh c)")
        self.permute_Wi_col = Rearrange("... c n_i n_im1 -> ... n_im1 c n_i")
        self.permute_Wip1 = Rearrange("... c n_ip1 n_i -> ... n_ip1 c n_i")
        self.ablate_crossterm = ablate_crossterm
        self.ablate_diagonalterm = ablate_diagonalterm

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights = [torch.zeros_like(w) for w in wsfeat.weights]
        out_biases = [torch.zeros_like(b) for b in wsfeat.biases]
        if self.share_projections:
            qkv = wsfeat.map(self.to_qkv)
        else:
            qkv_weights, qkv_biases = [], []
            for i in range(len(self.network_spec)):
                qkv_weights.append(self.weight_to_qkv[i](wsfeat.weights[i]))
                qkv_biases.append(self.bias_to_qkv[i](wsfeat.biases[i]))
            qkv = WeightSpaceFeatures(qkv_weights, qkv_biases)
        qkv = qkv.map(self.split_heads)
        rowcol_means = [w.mean(dim=(-2, -1)) for w in qkv.weights]  # (B, nh, c)
        bias_means = [b.mean(dim=-1) for b in qkv.biases]  # (B, nh, c)
        qkv_avgs = torch.stack(rowcol_means + bias_means, dim=-2)  # (B, nh, 2 * n_layers, 3ch)
        q_avg, k_avg, v_avg = qkv_avgs.tensor_split(3, dim=-1)  # (B, nh, 2 * n_layers, ch)
        if not self.ablate_diagonalterm:
            wb_out = self.combine_nh_nc(simple_attention(q_avg, k_avg, v_avg))
            w_out, b_out = wb_out.tensor_split(2, dim=-2)  # (B, n_layers, nh*ch)
            for i in range(len(out_weights)):
                w_out_i = w_out[:, i].unsqueeze(-1).unsqueeze(-1)
                if not self.share_projections: w_out_i = self.unproject_weight[i](w_out_i)
                out_weights[i] += w_out_i
                out_biases[i] += b_out[:, i].unsqueeze(-1)
        if self.ablate_crossterm:
            return unshape_wsfeat_symmetry(WeightSpaceFeatures(tuple(out_weights), tuple(out_biases)), self.network_spec)
        for i in range(-1, len(wsfeat)):
            inp = []
            if i > -1:
                n_i, n_im1 = wsfeat.weights[i].shape[-2], wsfeat.weights[i].shape[-1]
                Wi_cols = self.permute_Wi_col(qkv.weights[i])
                vi = qkv.biases[i].unsqueeze(-3)  # (B, nh, 1, c, n_i)
                inp.extend([Wi_cols, vi])
            if i < len(wsfeat) - 1:
                n_i = wsfeat.weights[i+1].shape[-1]
                inp.append(self.permute_Wip1(qkv.weights[i+1]))
            inp = torch.cat(inp, dim=-3)  # (B, nh, n_im1 + 1 + n_ip1, 3c, n_i)
            q_i, k_i, v_i = inp.tensor_split(3, dim=-2) # (B, nh, n_im1 + 1 + n_ip1, c, n_i)
            # TODO: can this be simplified?
            q_i = torch.flatten(q_i, start_dim=-2)
            k_i = torch.flatten(k_i, start_dim=-2)
            v_i = torch.flatten(v_i, start_dim=-2)
            # (B, nh, n_im1 + 1 + n_ip1, c * n_i)
            out = simple_attention(q_i, k_i, v_i, dropout=self.dropout)
            # ... (ch n_i) -> ... ch n_i
            out = out.view(*out.shape[:-1], self.ch, n_i)
            idx = 0
            if i > -1:
                # b nh n_im1 ch n_i -> b (nh ch) n_i n_im1
                out_weights_i = torch.flatten(out[:, :, :n_im1].permute(0, 1, 3, 4, 2), start_dim=1, end_dim=2)
                if not self.share_projections: out_weights_i = self.unproject_weight[i](out_weights_i)
                out_weights[i] += out_weights_i
                # b nh 1 ch n_i -> b (nh ch) n_i
                out_biases[i] += torch.flatten(out[:, :, n_im1: n_im1 + 1].squeeze(2), start_dim=1, end_dim=2)
                idx = n_im1 + 1
            if i < len(wsfeat) - 1:
                # b nh n_ip1 ch n_i -> b (nh ch) n_ip1 n_i
                out_weights_ip1 = torch.flatten(out[:, :, idx:].transpose(2, 3), start_dim=1, end_dim=2)
                if not self.share_projections: out_weights_ip1 = self.unproject_weight[i+1](out_weights_ip1)
                out_weights[i+1] += out_weights_ip1
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(tuple(out_weights), tuple(out_biases)), self.network_spec)

    def __repr__(self):
        return f"NPAttention(channels={self.ch * self.nh}, num_heads={self.nh}, dropout={self.dropout.p})"



class HNPS_SirenLinear(nn.Module):
    def __init__(self, in_network_spec: NetworkSpec, out_network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.in_network_spec = in_network_spec
        self.out_network_spec = out_network_spec
        layer_weight_shapes = in_network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(in_network_spec)
        in_filter_facs = [int(np.prod(spec.shape[2:])) for spec in in_network_spec.weight_spec]
        out_filter_facs = [int(np.prod(spec.shape[2:])) for spec in out_network_spec.weight_spec]
        
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

        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.out_network_spec)

class HNPSLinear(nn.Module):
    def __init__(self, in_network_spec: NetworkSpec, out_network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.in_network_spec = in_network_spec
        self.out_network_spec = out_network_spec

        layer_weight_shapes = in_network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(in_network_spec)
        in_filter_facs = [int(np.prod(spec.shape[2:])) for spec in in_network_spec.weight_spec]
        out_filter_facs = [int(np.prod(spec.shape[2:])) for spec in out_network_spec.weight_spec]
        
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

    
    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
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
                
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.out_network_spec)




class HNPSMixerLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.d = in_channels
        self.e = out_channels
        self.network_spec = network_spec
        layer_weight_shapes = network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(network_spec)
        self.L = L
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        
        #TODO: edit requires_grad
        self.Psi_s_L_t = nn.ParameterDict({f'{s}_{t}': nn.Parameter(torch.randn((self.d, 1, layer_out_weight_shape[-2]))/((self.d*layer_out_weight_shape[-2])), requires_grad=False) for s in range(1, L + 1) for t in range(0, s) })        
        self.Psi_s_0_L_t = nn.ParameterDict({f'{s}_{t}': nn.Parameter(torch.randn((self.d, layer_in_weight_shape[-1], layer_out_weight_shape[-2]))/((self.d*layer_in_weight_shape[-1]*layer_out_weight_shape[-2])), requires_grad=False) for s in range(1, L +1) for t in range(0, s) })
        
        self.Psi_s_0_L_s = nn.ParameterDict({f'{s}_{s}': nn.Parameter(torch.randn((self.d, layer_in_weight_shape[-1], layer_out_weight_shape[-2]))/((self.d*layer_in_weight_shape[-1]*layer_out_weight_shape[-2])), requires_grad=False) for s in range(1,L)})
        self.Psi_s_L_s =nn.ParameterDict({f'{s}_{s}': nn.Parameter(torch.randn((self.d, 1, layer_out_weight_shape[-2]))/((self.d*layer_out_weight_shape[-2])), requires_grad=False) for s in range(1,L)})      
        
        for i in range(L):
            fac_i = filter_facs[i]
            # define layer name as:
            #   layer_{i: layer index}_{W/b: output}_{W/b/Wb/bW/WW/none: input (none for bias term)_{index of input layer (if necessary)}}
            # example bias term: layer_0_W
            if i == 0:
                self.layer_0_W_W = EinsumLayer(equation="edqk, bdjq -> bejk",
                                              weight_shape=[fac_i * self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_W_WW = EinsumLayer(equation="edqk, bdjq -> bejk",
                                              weight_shape=[fac_i * self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_W_bW = EinsumLayer(equation="edqk, bdjq -> bejk",
                                              weight_shape=[fac_i * self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_W_b = EinsumLayer(equation="edk, bdj -> bejk",
                                               weight_shape=[fac_i * self.e, self.d,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])

                self.layer_0_b_W = EinsumLayer(equation="edq, bdjq -> bej",
                                               weight_shape=[self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_b_WW = EinsumLayer(equation="edq, bdjq -> bej",
                                               weight_shape=[self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_b_bW = EinsumLayer(equation="edq, bdjq -> bej",
                                               weight_shape=[self.e, fac_i * self.d,
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_b_b = EinsumLayer(equation="ed, bdj -> bej",
                                               weight_shape=[self.e, self.d],
                                               fan_in_mask=[0, 1])
                set_init_einsum_(
                    self.layer_0_W_W,
                    self.layer_0_W_WW,
                    self.layer_0_W_bW,
                    self.layer_0_W_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_b_W,
                    self.layer_0_b_WW,
                    self.layer_0_b_bW,
                    self.layer_0_b_b,
                    init_type=init_type,
                )

            elif i == L-1:
                self.add_module(f"layer_{i}_W_W", EinsumLayer(equation="edpj, bdpk -> bejk",
                                            weight_shape=[fac_i * self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0]))
                self.add_module(f"layer_{i}_W_WW", EinsumLayer(equation="edpj, bdpk -> bejk",
                                            weight_shape=[fac_i * self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0]))
                self.add_module(f"layer_{i}_W_bW", EinsumLayer(equation="edpj, bdpk -> bejk",
                                            weight_shape=[fac_i * self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0]))

                self.add_module(f"layer_{i}_b_WW_L0L0", EinsumLayer(equation="edpqj, bdpq -> bej",
                                            weight_shape=[self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_in_weight_shape[-1], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 1, 0]))
                self.add_module(f"layer_{i}_b_W", EinsumLayer(equation="edpqj, bdpq -> bej",
                                            weight_shape=[self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_in_weight_shape[-1], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 1, 0]))
                self.add_module(f"layer_{i}_b_bW_LL0", EinsumLayer(equation="edpqj, bdpq -> bej",
                                            weight_shape=[self.e, fac_i * self.d,
                                                          layer_out_weight_shape[-2], layer_in_weight_shape[-1], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 1, 0]))
                self.add_module(f"layer_{i}_b_Wb", EinsumLayer(equation="edpj, bdp -> bej",
                                            weight_shape=[self.e,  self.d,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0]))
                
                self.add_module(f"layer_{i}_b_WW_0L", EinsumLayer(equation="edlj, bdl -> bej",
                                weight_shape=[self.e,  self.d,
                                                self.L-1, layer_out_weight_shape[-2]],
                                fan_in_mask=[0, 1, 1, 0]))
                self.add_module(f"layer_{i}_b_bW_L", EinsumLayer(equation="edlj, bdl -> bej",
                                                weight_shape=[self.e, self.d,
                                                            self.L-1, layer_out_weight_shape[-2]],
                                                fan_in_mask=[0, 1, 1, 0]))
                
                self.add_module(f"layer_{i}_b_b_L", EinsumLayer(equation="edpj, bdp -> bej",
                                            weight_shape=[self.e,  self.d,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0]))
                self.add_module(f"layer_{i}_b", EinsumLayer(equation="ej, b-> bej",
                                            weight_shape=[self.e, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0]))
                set_init_einsum_(
                    getattr(self, f"layer_{i}_W_W"),
                    getattr(self, f"layer_{i}_W_WW"),
                    getattr(self, f"layer_{i}_W_bW"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_b_WW_L0L0"),
                    getattr(self, f"layer_{i}_b_W"),
                    getattr(self, f"layer_{i}_b_bW_LL0"),
                    getattr(self, f"layer_{i}_b_Wb"),
                    getattr(self, f"layer_{i}_b_b_L"),
                    getattr(self, f"layer_{i}_b"),
                    getattr(self, f"layer_{i}_b_WW_0L"),
                    getattr(self, f"layer_{i}_b_bW_L"),
                    init_type=init_type,
                )

            else:
                # Case where 1 < i < L
                self.add_module(f"layer_{i}_W_W", EinsumLayer(
                    equation="ed, bdjk -> bejk",
                    weight_shape=[self.e * fac_i, self.d * fac_i],
                    fan_in_mask=[0, 1]
                ))
                self.add_module(f"layer_{i}_W_WW", EinsumLayer(
                    equation="ed, bdjk -> bejk",
                    weight_shape=[self.e * fac_i, self.d * fac_i],
                    fan_in_mask=[0, 1]
                ))
                self.add_module(f"layer_{i}_W_bW", EinsumLayer(
                    equation="ed, bdjk -> bejk",
                    weight_shape=[self.e * fac_i, self.d * fac_i],
                    fan_in_mask=[0, 1]
                ))

                set_init_einsum_(
                    getattr(self, f"layer_{i}_W_W"),
                    getattr(self, f"layer_{i}_W_WW"),
                    getattr(self, f"layer_{i}_W_bW"),
                    init_type=init_type,
                )

                self.add_module(f"layer_{i}_b_W", EinsumLayer(
                    equation="edq, bdjq -> bej",
                    weight_shape=[self.e, self.d * fac_i, layer_in_weight_shape[-1]],
                    fan_in_mask=[0, 1, 1]
                ))
                self.add_module(f"layer_{i}_b_WW", EinsumLayer(
                    equation="edq, bdjq -> bej",
                    weight_shape=[self.e, self.d * fac_i, layer_in_weight_shape[-1]],
                    fan_in_mask=[0, 1, 1]
                ))
                self.add_module(f"layer_{i}_b_Wb", EinsumLayer(
                    equation="ed, bdj -> bej",
                    weight_shape=[self.e, self.d],
                    fan_in_mask=[0, 1]
                ))
                self.add_module(f"layer_{i}_b_bW", EinsumLayer(
                    equation="edq, bdjq -> bej",
                    weight_shape=[self.e, self.d * fac_i, layer_in_weight_shape[-1]],
                    fan_in_mask=[0, 1, 1]
                ))
                self.add_module(f"layer_{i}_b_b", EinsumLayer(
                    equation="ed, bdj -> bej",
                    weight_shape=[self.e, self.d],
                    fan_in_mask=[0, 1]
                ))

                set_init_einsum_(
                    getattr(self, f"layer_{i}_b_W"),
                    getattr(self, f"layer_{i}_b_WW"),
                    getattr(self, f"layer_{i}_b_Wb"),
                    getattr(self, f"layer_{i}_b_bW"),
                    getattr(self, f"layer_{i}_b_b"),
                    init_type=init_type,
                )
    def mix_layers(self, wsfeat):
        L = len(self.network_spec)

        W = {i + 1: wsfeat[i][0] for i in range(L)}
        b = {i + 1: wsfeat[i][1] for i in range(L)}

        W_st = {}
        Wb_st = {}
        bW_st = {}
        WW_st = {}

        device = W[1].device
        d = self.d
        batch_size = W[1].shape[0]

        # Compute Psi transformations for all s
        #bs: n_s x 1 
        #W_L: n_L x n_L-1

    
        # Compute matrices for all s, t where s > t
        for s in range(1, L + 1):
            for t in range(0, s):
                # Compute W_{s,t}
                if t == s - 1:
                    W_st[(s, t)] = W[s]
                else:
                    #W_st[(s, t)] = torch.einsum('bdij,bdjk->bdik', W[s], W_st[(s - 1, t)])
                    W_st[(s,t)] = W[s] @ W_st[(s - 1, t)]
                assert W_st[(s, t)].shape == (batch_size, d, W[s].shape[-2], W[t + 1].shape[-1]), f"Shape mismatch for W_st[{s},{t}]"

                # Compute [Wb]_{(s, t)(t)}
                if t > 0:
                    #Wb_st[(s, t)] = torch.einsum('bdij,bdj->bdi', W_st[(s, t)], b[t])
                    Wb_st[(s, t)] = torch.matmul(W_st[(s, t)], b[t].unsqueeze(-1)).squeeze(-1) # matrix vector multiplication
                    assert Wb_st[(s, t)].shape == (batch_size, d, W[s].shape[-2]), f"Shape mismatch for Wb_st[{s},{t}]"

        for s in range(1, L + 1):
            for t in range(0, s):
                # Compute [bW]^{(s)}(L, t)
                bW_st[(s,t)] = b[s].unsqueeze(-1) @ self.Psi_s_L_t[f'{s}_{t}'].unsqueeze(0) @ W_st[(L, t)]
                assert bW_st[(s, t)].shape == (batch_size, d, W[s].shape[-2], W[t + 1].shape[-1]), f"Shape mismatch for bW_st[{s},{t}]"
                
                # Compute [WW]^{(s, 0)}(L, t)
                WW_st[(s,t)] = W_st[(s, 0)] @ self.Psi_s_0_L_t[f'{s}_{t}'] @ W_st[(L, t)]
                assert WW_st[(s, t)].shape == (batch_size, d, W[s].shape[-2], W[t + 1].shape[-1]), f"Shape mismatch for WW_st[{s},{t}]"
        
        WW_pp  = torch.empty((batch_size, d, L-1), device=device)
        bW_pp = torch.empty((batch_size, d, L-1), device=device)
        for s in range(1,L):
            WW_st[(s,s)] = W_st[(s, 0)] @ self.Psi_s_0_L_s[f'{s}_{s}'] @ W_st[(L, s)] # B D n_s n_s -> B D
            # (torch.einsum('bdnn->bd',WW_st[(s,s)])) #compute trace over the last two dimensions
            WW_pp[:,:,s-1] = (torch.einsum('bdnn->bd',WW_st[(s,s)]))
            
            bW_st[(s,s)] = b[s].unsqueeze(-1) @ self.Psi_s_L_s[f'{s}_{s}'].unsqueeze(0) @ W_st[(L, s)]
            bW_pp[:,:,s-1] = (torch.einsum('bdnn->bd',bW_st[(s,s)]))
        
        return W_st, Wb_st, bW_st, WW_st, b, WW_pp, bW_pp


    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights, out_biases = [], []
        
        L = len(self.network_spec)

        W_st, Wb_st, bW_st, WW_st, b, WW_pp, bW_pp = self.mix_layers(wsfeat)

        for i in range(0, L):
            weight, bias = wsfeat[i]
            if  i == 0:
                W_W_term  =   self.layer_0_W_W(W_st[(1,0)])
                W_WW_term =     self.layer_0_W_WW(WW_st[(1, 0)])
                W_bW_term = self.layer_0_W_bW(bW_st[(1, 0)])
                W_b_term  =     self.layer_0_W_b(b[1])
                E_W = W_bW_term + W_W_term + W_b_term + W_WW_term   
                out_weights.append(E_W)
                
                b_W_term = self.layer_0_b_W(W_st[(1,0)])
                b_WW_term = self.layer_0_b_WW(WW_st[(1, 0)])
                b_bW_term = self.layer_0_b_bW(bW_st[(1, 0)])
                b_b_term = self.layer_0_b_b(b[1])
                
                E_b = b_W_term +  b_b_term +b_WW_term + b_bW_term 
                out_biases.append(E_b)
            elif i == L-1:
                W_W_term = getattr(self, f"layer_{i}_W_W")(W_st[(i+1, i)])
                W_WW_term = getattr(self, f"layer_{i}_W_WW")(WW_st[(i+1, i)])
                W_bW_term = getattr(self, f"layer_{i}_W_bW")(bW_st[(i+1, i)])

                E_W = W_W_term + W_bW_term + W_WW_term
                out_weights.append(E_W)
                
                b_WW_L0L0_term = getattr(self, f"layer_{i}_b_WW_L0L0")(WW_st[(i+1, 0)])
                b_W_term = getattr(self, f"layer_{i}_b_W")(W_st[(i+1, 0)])
                b_bW_LL0_term = getattr(self, f"layer_{i}_b_bW_LL0")(bW_st[(i+1, 0)])
                
                b_b_L_term = getattr(self, f"layer_{i}_b_b_L")(bias)
                b_term = getattr(self, f"layer_{i}_b")(torch.tensor([1], device=weight.device))
                
                b_Wb_st_term = sum(getattr(self, f"layer_{i}_b_Wb")(Wb_st[(i+1, s)]) for s in range(1,L))
                b_WW_pp_term = getattr(self, f"layer_{i}_b_WW_0L")(WW_pp)
                b_bW_pp_term = getattr(self, f"layer_{i}_b_bW_L")(bW_pp)
                E_b = b_b_L_term  + b_term +b_W_term +b_WW_L0L0_term + b_bW_LL0_term + b_Wb_st_term + b_WW_pp_term + b_bW_pp_term
                out_biases.append(E_b)

            else:
                # For the weight E(W)^{(i)}_{jk}
                W_W_term = getattr(self, f"layer_{i}_W_W")(W_st[(i+1,i)])
                W_WW_term = getattr(self, f"layer_{i}_W_WW")(WW_st[(i+1, i)])
                W_bW_term = getattr(self, f"layer_{i}_W_bW")(bW_st[(i+1, i)])

                # Sum all terms to get E(W)^{(i)}_{jk}
                E_W = W_W_term + W_bW_term + W_WW_term 
                out_weights.append(E_W)

                # For the bias E(b)^{(i)}_{j}
                b_W_term = getattr(self, f"layer_{i}_b_W")(W_st[(i+1, 0)])
                b_WW_term = getattr(self, f"layer_{i}_b_WW")(WW_st[(i+1, 0)])
                b_Wb_term = sum(getattr(self, f"layer_{i}_b_Wb")(Wb_st[(i+1, t)]) for t in range(1, i+1))
                b_bW_term = getattr(self, f"layer_{i}_b_bW")(bW_st[(i+1, 0)])
                b_b_term = getattr(self, f"layer_{i}_b_b")(b[i+1])

                # Sum all terms to get E(b)^{(i)}_{j}
                E_b = b_b_term + b_Wb_term + b_W_term + b_WW_term + b_bW_term 
                out_biases.append(E_b)

        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)
        #return wsfeat

class EinsumLayer(nn.Module):
    def __init__(self, equation="", weight_shape=None, fan_in_mask=None) -> None:
        super().__init__()
        self.equation = equation
        if len(self.equation) == 0:
            return

        self.weight_shape_list = weight_shape
        self.weight_shape_tensor = torch.tensor(weight_shape, dtype=torch.int)
        self.fan_in_mask = torch.tensor(fan_in_mask).ge(0.5)
        if torch.all(self.fan_in_mask == False):
            self.fan_in = 0
        else:
            self.fan_in = torch.prod(self.weight_shape_tensor[self.fan_in_mask])
        self.fan_out_mask = torch.tensor(fan_in_mask).lt(0.5)
        if torch.all(self.fan_out_mask == False):
            self.fan_out = 0
        else:
            self.fan_out = torch.prod(self.weight_shape_tensor[self.fan_out_mask])

        self.equation = equation
        self.weight = nn.Parameter(torch.empty(self.weight_shape_list))
        #self.weight = nn.Parameter(torch.ones(self.weight_shape_list))

    def forward(self, input):
        if len(self.equation) > 0:
            return torch.einsum(self.equation, self.weight, input)