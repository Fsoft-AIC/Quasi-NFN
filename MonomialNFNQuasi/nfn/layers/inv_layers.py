import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nfn.common import NetworkSpec, WeightSpaceFeatures
from nfn.layers.layer_utils import (
    set_init_,
    set_init_einsum_,
    shape_wsfeat_symmetry,
    unshape_wsfeat_symmetry,
)
from nfn.layers.equiv_layers import EinsumLayer
from nfn.layers.mlp_quasi import SimpleMLP
class NPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out.append(weight.mean(dim=(2,3)).unsqueeze(-1))
            out.append(bias.mean(dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        num_outs = 0
        for fac in filter_facs:
            num_outs += fac + 1
        return num_outs


class HNPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            if i == 0:
                out.append(weight.mean(dim=2))  # average over rows
            elif i == len(wsfeat) - 1:
                out.append(weight.mean(dim=3))  # average over cols
            else:
                out.append(weight.mean(dim=(2,3)).unsqueeze(-1))
            if i == len(wsfeat) - 1: out.append(bias)
            else: out.append(bias.mean(dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs


class HNPSNormalize(nn.Module):
    def __init__(self, network_spec: NetworkSpec, nfn_channels, mode_normalize="param_mul_L2"):
        super().__init__()
        self.network_spec = network_spec
        self.mode_normalize = mode_normalize
        self.nfn_channels = nfn_channels
        if self.mode_normalize == "param_mul_L2":
            for i in range(len(network_spec)):
                if len(network_spec.weight_spec[i].shape) == 4: #CNN
                    self.add_module(f"regularize_W_{i}",
                                    ElementwiseParamNormalize(nfn_channels *
                                                math.prod(network_spec.weight_spec[i].shape[-2:]),
                                                mode_normalize=mode_normalize)
                                    )
                else:
                    self.add_module(f"regularize_W_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_normalize))
                self.add_module(f"regularize_b_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_normalize))


    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out_weights = []
        out_biases = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            if self.mode_normalize == "param_mul_L2":
                regularizer_w = getattr(self, f"regularize_W_{i}")
                regularizer_b = getattr(self, f"regularize_b_{i}")
            else:
                regularizer_w = self.regularize_without_param
                regularizer_b = self.regularize_without_param


            if i == 0:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)

            elif i == len(wsfeat) - 1:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)

            else:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)

            # bias_regularized = F.normalize(bias, dim=1, p=2.0)
            bias_regularized = regularizer_b(bias)
            out_biases.append(bias_regularized)

        return WeightSpaceFeatures(out_weights, out_biases)

    def regularize_without_param(self, weight):
        if self.mode_normalize == "L1":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=1.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_normalize == "L2":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_normalize == "L2_square":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2

        return weight_regularized

class HNPSPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec, nfn_channels, mode_pooling="param_mul_L2"):
        super().__init__()
        self.network_spec = network_spec
        self.mode_pooling = mode_pooling
        self.nfn_channels = nfn_channels
        if self.mode_pooling == "param_mul_L2":
            for i in range(len(network_spec)):
                if len(network_spec.weight_spec[i].shape) == 4: #CNN
                    self.add_module(f"regularize_W_{i}",
                                    ElementwiseParamNormalize(nfn_channels *
                                                math.prod(network_spec.weight_spec[i].shape[-2:]),
                                                mode_normalize=mode_pooling)
                                    )
                else:
                    self.add_module(f"regularize_W_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))
                self.add_module(f"regularize_b_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))


    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            if self.mode_pooling == "param_mul_L2":
                regularizer_w = getattr(self, f"regularize_W_{i}")
                regularizer_b = getattr(self, f"regularize_b_{i}")
            else:
                regularizer_w = self.regularize_without_param
                regularizer_b = self.regularize_without_param


            if i == 0:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=2))  # average over rows

            elif i == len(wsfeat) - 1:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=3))  # average over cols

            else:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=(2,3)).unsqueeze(-1))

            if i == len(wsfeat) - 1:
                out.append(bias)
            else:
                # bias_regularized = F.normalize(bias, dim=1, p=2.0)
                bias_regularized = regularizer_b(bias)
                out.append(bias_regularized.mean(dim=-1).unsqueeze(-1))

        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    def regularize_without_param(self, weight):
        if self.mode_pooling == "L1":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=1.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2_square":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2

        return weight_regularized

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs

class HNPSMixerInv(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.d = in_channels
        self.e = out_channels
        self.network_spec = network_spec
        layer_weight_shapes = network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        L = len(network_spec)
        self.L = L
        
        self.Psi_s_L_t = nn.Parameter(torch.randn((self.d, 1, layer_out_weight_shape[-2]))/(self.d*layer_out_weight_shape[-2]), requires_grad=True)      
        self.Psi_s_0_L_t = nn.Parameter(torch.randn((self.d, layer_in_weight_shape[-1], layer_out_weight_shape[-2]))/(self.d*layer_in_weight_shape[-1]*layer_out_weight_shape[-2]), requires_grad=True)
        
        self.Psi_s_0_L_s = nn.ParameterDict({f'{s}_{s}': nn.Parameter(torch.randn((self.d, layer_in_weight_shape[-1], layer_out_weight_shape[-2]))/(self.d*layer_in_weight_shape[-1]*layer_out_weight_shape[-2]), requires_grad=True) for s in range(1,L)})
        self.Psi_s_L_s =nn.ParameterDict({f'{s}_{s}': nn.Parameter(torch.randn((self.d, 1, layer_out_weight_shape[-2]))/(self.d*layer_out_weight_shape[-2]), requires_grad=True) for s in range(1,L)})      


            # define layer name as:
            #   layer_{i: layer index}_{W/b: output}_{W/b/Wb/bW/WW/none: input (none for bias term)_{index of input layer (if necessary)}}
            # example bias term: layer_0_W
        self.layer_WW = EinsumLayer(equation="edpq, bdpq -> be",
                                        weight_shape=[ self.e,  self.d,
                                                        layer_out_weight_shape[-2], layer_in_weight_shape[-1]],
                                        fan_in_mask = [0, 1, 1, 1])
        self.layer_W = EinsumLayer(equation="edpq, bdpq -> be",
                                        weight_shape=[ self.e, self.d,
                                                        layer_out_weight_shape[-2], layer_in_weight_shape[-1]],
                                        fan_in_mask = [0, 1, 1, 1])
        self.layer_bW = EinsumLayer(equation="edpq, bdpq -> be",
                                        weight_shape=[ self.e,  self.d,
                                                        layer_out_weight_shape[-2], layer_in_weight_shape[-1]],
                                        fan_in_mask = [0, 1, 1, 1])
        self.layer_Wb =  EinsumLayer(equation="edp, bdp -> be",
                                        weight_shape=[self.e,  self.d,
                                                        layer_out_weight_shape[-2]],
                                        fan_in_mask=[0, 1, 1])
        self.layer_WW_pp = EinsumLayer(equation="edl, bdl -> be",
                                        weight_shape=[self.e,  self.d,
                                                        self.L-1],
                                        fan_in_mask=[0, 1, 1])
        self.layer_bW_pp = EinsumLayer(equation="edl, bdl -> be",
                                        weight_shape=[self.e,  self.d,
                                                        self.L-1],
                                        fan_in_mask=[0, 1, 1])
        self.layer_b = EinsumLayer(equation="edp, bdp -> be",
                                        weight_shape=[self.e,  self.d,
                                                        layer_out_weight_shape[-2]],
                                        fan_in_mask=[0, 1, 1])
        self.tau_weight = nn.Parameter(torch.rand((self.e)))
        layers = [self.layer_WW, self.layer_W, self.layer_bW, self.layer_Wb, \
            self.layer_WW_pp, self.layer_bW_pp, self.layer_b]
        set_init_einsum_(*layers, init_type=init_type)
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

        # Compute matrices for all s, t where s = L
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

        bW_st[(L,0)] = b[L].unsqueeze(-1) @ self.Psi_s_L_t.unsqueeze(0) @ W_st[(L, 0)]
        assert bW_st[(L,0)].shape == (batch_size, d, W[L].shape[-2], W[1].shape[-1]), f"Shape mismatch for bW_st[{L},{0}]"
        
        # Compute [WW]^{(s, 0)}(L, t)
        WW_st[(L,0)] = W_st[(L, 0)] @ self.Psi_s_0_L_t @ W_st[(L, 0)]
        assert WW_st[(L,0)].shape == (batch_size, d, W[L].shape[-2], W[1].shape[-1]), f"Shape mismatch for WW_st[{L},{0}]"
        
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
        
        L = len(self.network_spec)

        W_st, Wb_st, bW_st, WW_st, b, WW_pp, bW_pp = self.mix_layers(wsfeat)

        WW_term = self.layer_WW(WW_st[(L,0)])
        W_term = self.layer_W(W_st[(L,0)])
        WW_pp_term = self.layer_WW_pp(WW_pp)
        bW_pp_term = self.layer_bW_pp(bW_pp)
        bW_term = self.layer_bW(bW_st[(L,0)])
        Wb_term = sum([self.layer_Wb(Wb_st[(L, t)]) for t in range(1, L)])
        b_term = self.layer_b(b[L])
        tau_term = torch.ones((wsfeat[0][0].shape[0]), device =wsfeat[0][0].device).unsqueeze(1) @ self.tau_weight.unsqueeze(0)
        out = WW_term + W_term + WW_pp_term + bW_pp_term + bW_term + Wb_term + b_term + tau_term
        return torch.flatten(out,start_dim=-1)
        #return wsfeat



class ElementwiseParamNormalize(nn.Module):
    def __init__(self, hidden, mode_normalize) -> None:
        super().__init__()
        self.hidden = hidden
        self.mode_normalize = mode_normalize
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.ones(hidden))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)


    def forward(self, input):
        if self.mode_normalize == "param_mul_L2":
            if input.dim() == 6: #C NN
                    input_shape = input.shape
                    input = rearrange(input, 'b c i j k l -> b i j (c k l)')
                    input_regularized = F.normalize(input, p=2.0, dim=-1)
                    input_regularized = self.weight * input_regularized + self.bias
                    input_regularized = rearrange(input_regularized, 'b i j (c k l) -> b c i j k l',
                                                    c = input_shape[1], k = input_shape[-2],
                                                    l = input_shape[-1])
            elif input.dim() == 4: # MLP
                input = rearrange(input, 'b c i j-> b i j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b i j c -> b c i j')

            elif input.dim() == 3: #bias
                input = rearrange(input, 'b c j-> b j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b j c -> b c j')
        return input_regularized
