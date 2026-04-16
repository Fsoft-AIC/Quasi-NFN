import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from nfn_transformer.common.weight_space import AttentionNetworkSpec, AttentionWeightSpaceFeatures, NetworkSpec, LinearWeightSpaceFeatures
from nfn_transformer.layers.layer_utils import (
    set_init_,
    set_init_einsum_,
)

from nfn_transformer.layers import EinsumLayer

class GLQuasiAlpha(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # Use aggregated statistics instead of full flattening
            # Features: mean, std, min, max of weights and biases
            
            input_dim = (3 * ( h * D * D_q) +   h * D_v * D +  D * D_a +  D_a +  D_a * D +  D ) * self.L



            self.add_module(f"layer_{i}_M", 
                StableMLPGL(input_dim=input_dim, output_dim=D_k, n_h = h, hidden_dims=32, mode = 'exp'))
            
            self.add_module(f"layer_{i}_N", 
                StableMLPGL(input_dim=input_dim, output_dim=D_v, n_h = h, hidden_dims=32, mode = 'exp'))
    
    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        all_M = []
        all_N = []
        L = len(wsfeat)  # Number of layers
        all_flat_input = []
        # Loop over each layer's weights and biases
        for i in range(L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]
            flat_input = torch.cat([W_q.flatten(start_dim=2), W_k.flatten(start_dim=2), W_v.flatten(start_dim=2), W_o.flatten(start_dim=2),\
                                    W_A.flatten(start_dim=2),W_B.flatten(start_dim=2),b_A.flatten(start_dim=2),b_B.flatten(start_dim=2),], dim=-1)
            all_flat_input.append(flat_input)
        all_flat_input = torch.cat(all_flat_input, dim=-1)

        for i in range(L):

            M = getattr(self, f"layer_{i}_M")(all_flat_input)
            N = getattr(self, f"layer_{i}_N")(all_flat_input)
            all_M.append(M)
            all_N.append(N)
        
        return all_M, all_N
    
class GLQuasiAlphaPer(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # Use aggregated statistics instead of full flattening
            # Features: mean, std, min, max of weights and biases
            
            input_dim = 3 * ( h * D * D_q) +   h * D_v * D +  D * D_a +  D_a +  D_a * D +  D 


            self.add_module(f"layer_{i}_norm", nn.LayerNorm(input_dim))

            self.add_module(f"layer_{i}_M", 
                StableMLPGL(input_dim=input_dim, output_dim=D_k, n_h = h, hidden_dims=32, mode = 'jitter'))
            
            self.add_module(f"layer_{i}_N", 
                StableMLPGL(input_dim=input_dim, output_dim=D_v, n_h = h, hidden_dims=32, mode = 'jitter'))

    
    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        all_M = []
        all_N = []
        L = len(wsfeat)  # Number of layers
        # Loop over each layer's weights and biases
        for i in range(L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]
            flat_input = torch.cat([W_q.flatten(start_dim=2), W_k.flatten(start_dim=2), W_v.flatten(start_dim=2), W_o.flatten(start_dim=2),\
                                    W_A.flatten(start_dim=2),W_B.flatten(start_dim=2),b_A.flatten(start_dim=2),b_B.flatten(start_dim=2),], dim=-1)
            flat_input = getattr(self, f"layer_{i}_norm")(flat_input)

            M = getattr(self, f"layer_{i}_M")(flat_input)
            N = getattr(self, f"layer_{i}_N")(flat_input)
            all_M.append(M)
            all_N.append(N)
        
        return all_M, all_N


class GLQuasiAlphaPooled(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # Use aggregated statistics instead of full flattening
            # Features: mean, std, min, max of weights and biases
            
            input_dim = 7 * 8 * self.L # 7 features, 8 terms
            # self.add_module(f"layer_{i}_norm", nn.LayerNorm(7 * 8))
            self.add_module(f"layer_{i}_M", 
            SimpleMLPGL(input_dim=input_dim, output_dim=D_k, n_h = h, hidden_dims=[32], mode="jitter"))
            
            self.add_module(f"layer_{i}_N", 
                SimpleMLPGL(input_dim=input_dim, output_dim=D_v, n_h = h, hidden_dims=[32], mode="jitter"))
    
    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        all_M = []
        all_N = []
        L = len(wsfeat)  # Number of layers
    
        # Loop over each layer's weights and biases
        out = []
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)

        for i in range(L):
            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)  # (5, B, C)
                feats.append(torch.stack([mean, var, *q], dim=-1))  # (B, C, 7)
            out.append(torch.cat(feats, dim=-1))  # concat all terms at layer i

        out = torch.cat(out, dim=-1)  # (B, C, 7*L*8)
        # out = F.layer_norm(out, out.shape[-1:])    
           
        for i in range(L):
            M = getattr(self, f"layer_{i}_M")(out)
            N = getattr(self, f"layer_{i}_N")(out)
            all_M.append(M)
            all_N.append(N)

        return all_M, all_N

class GLQuasiAlphaPooledPerLayer(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # Use aggregated statistics instead of full flattening
            # Features: mean, std, min, max of weights and biases
            
            input_dim = 7 * 8  # 7 features, 8 terms
            self.add_module(f"layer_{i}_norm", nn.LayerNorm(input_dim))
            self.add_module(f"layer_{i}_M", 
                SimpleMLPGL(input_dim=input_dim, output_dim=D_k, n_h = h, hidden_dims=[32], mode="exp"))
            
            self.add_module(f"layer_{i}_N", 
                SimpleMLPGL(input_dim=input_dim, output_dim=D_v, n_h = h, hidden_dims=[32], mode="exp"))
    
    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        all_M = []
        all_N = []
        L = len(wsfeat)  # Number of layers
    
        # Loop over each layer's weights and biases
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)

        for i in range(L):
            out = []

            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)  # (5, B, C)
                feats.append(torch.stack([mean, var, *q], dim=-1))  # (B, C, 7)
            out.append(torch.cat(feats, dim=-1))  # concat all terms at layer i
            out = torch.cat(out, dim=-1)  # (B, C, 7*8)
            out = getattr(self, f"layer_{i}_norm")(out)
            M = getattr(self, f"layer_{i}_M")(out)
            N = getattr(self, f"layer_{i}_N")(out)
            all_M.append(M)
            all_N.append(N)


        return all_M, all_N

class SimpleMLPGL(nn.Module):
    def __init__(self, input_dim, output_dim, n_h, hidden_dims=[256, 256],
                 activation=nn.ReLU(), init_type="xavier", mode="exp"):
        super().__init__()
        self.mode = mode
        self.out_dim = output_dim
        self.n_h = n_h
        self.init_type = init_type
        final_out = output_dim * output_dim * n_h

        dims = [input_dim] + hidden_dims + [final_out]
        layers = []
        for i in range(len(dims) - 1):

            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # not last
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation)
        self.net = nn.Sequential(*layers)
        self.current_reg_loss = 0
        self._initialize_weights()
        self.scale_val = nn.Parameter(torch.tensor(0.01), requires_grad=True)  # exp(0)=1 initial
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
        B, C = x.shape[0], x.shape[1]
        raw = self.net(x)
        N = self.out_dim

        if self.mode == "exp":
            # reshape
            raw = raw.view(B, C, self.n_h, N, N)
            # small factor keeps exp close to I
            #small_factor = torch.sigmoid(self.small_factor) * (max_val - min_val) + min_val

            mat = torch.matrix_exp(self.scale_val * torch.sin(raw))

        elif self.mode == "jitter":
            raw = raw.view(B, C, self.n_h, N, N)
            # raw = raw * self.scale_val # scale to [scale_low, scale_high]
            eye = torch.eye(self.out_dim, device=raw.device).view(1, 1, self.out_dim, self.out_dim)
            mat = self.scale_val * torch.sin(raw) + eye  # small jitter on diagonal


        return mat
    
class StableMLPGL(nn.Module):
    def __init__(self, input_dim, output_dim, n_h,
                 hidden_dims=32, activation=nn.GELU(),
                 init_type="xavier", mode="lu"):
        super().__init__()
        self.mode = mode

        self.activation = activation
        self.out_dim = output_dim
        self.n_h = n_h
        N = output_dim
        if mode == "lu":
            n_L = N * (N - 1) // 2
            n_U = N * (N + 1) // 2
            final_out = (n_L + n_U) * n_h
        else:  # exp and jitter need full matrices
            final_out = N * N * n_h
        self.fc1 = nn.Linear(input_dim, hidden_dims)
        self.norm1 = nn.LayerNorm(hidden_dims)
        self.gate = nn.Linear(hidden_dims, hidden_dims)
        self.fc_out = nn.Linear(hidden_dims, final_out)


        # per-head global scale, initialized at identity (exp(0)=1)
        # self.scale_val = nn.Parameter(torch.zeros(n_h))
        self.small_factor = nn.Parameter(torch.tensor(0.1), requires_grad = True)  # start closer to identity
        self._initialize_weights(init_type)
        
    def _initialize_weights(self, init_type):
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

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        h = self.activation(self.norm1(self.fc1(x)))
        gated = h * torch.sigmoid(self.gate(h))  # gating for stability
        raw = self.fc_out(gated)
        min_val = 0.0001
        max_val = 0.01
        N = self.out_dim

        if self.mode == "exp":
            # reshape
            raw = raw.view(B, C, self.n_h, N, N)
            # small factor keeps exp close to I
            #small_factor = torch.sigmoid(self.small_factor) * (max_val - min_val) + min_val

            mat = torch.matrix_exp(self.small_factor * torch.sin(raw))
            #scale = torch.exp(self.scale_val).view(1, 1, self.n_h, 1, 1)
            # mat = scale * mat

        elif self.mode == "jitter":
            raw = raw.view(B, C, self.n_h, N, N)
            eye = torch.eye(N, device=raw.device).view(1, 1, 1, N, N)
            # start from identity, add small bounded perturbation
            perturb = self.small_factor * torch.tanh(raw)
            mat = eye + perturb
            #scale = torch.exp(self.scale_val).view(1, 1, self.n_h, 1, 1)
            # mat = scale * mat

        return mat
    
class StableMLPGL(nn.Module):
    def __init__(self, input_dim, output_dim, n_h,
                 hidden_dims=[256, 256], activation=nn.ReLU(),
                 init_type="xavier", mode="lu"):
        super().__init__()
        self.mode = mode
        self.out_dim = output_dim
        self.n_h = n_h
        self.init_type = init_type

        N = output_dim
        if mode == "lu":
            n_L = N * (N - 1) // 2
            n_U = N * (N + 1) // 2
            final_out = (n_L + n_U) * n_h
        else:  # exp and jitter need full matrices
            final_out = N * N * n_h

        # MLP backbone
        dims = [input_dim] + hidden_dims + [final_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # not last
                layers.append(nn.LayerNorm(dims[i+1]))

                layers.append(activation)
        self.net = nn.Sequential(*layers)

        # per-head global scale, initialized at identity (exp(0)=1)
        # self.scale_val = nn.Parameter(torch.zeros(n_h))
        self.small_factor = nn.Parameter(torch.tensor(0.01), requires_grad = False)  # start closer to identity
        self._initialize_weights()

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
        B, C = x.shape[0], x.shape[1]
        raw = self.net(x)

        N = self.out_dim

        if self.mode == "exp":
            # reshape
            raw = raw.view(B, C, self.n_h, N, N)
            # small factor keeps exp close to I
            mat = torch.matrix_exp(self.small_factor * raw)
            #scale = torch.exp(self.scale_val).view(1, 1, self.n_h, 1, 1)
            # mat = scale * mat

        elif self.mode == "jitter":
            raw = raw.view(B, C, self.n_h, N, N)
            eye = torch.eye(N, device=raw.device).view(1, 1, 1, N, N)
            # start from identity, add small bounded perturbation
            perturb = self.small_factor * torch.tanh(raw)
            mat = eye + perturb
            #scale = torch.exp(self.scale_val).view(1, 1, self.n_h, 1, 1)
            # mat = scale * mat

        elif self.mode == "lu":
            n_L = N * (N - 1) // 2
            n_U = N * (N + 1) // 2
            total = n_L + n_U
            raw = raw.view(B, C, self.n_h, total)

            mats = []
            for h in range(self.n_h):
                vec = raw[:, :, h, :]        # (B, C, total)
                l_part = vec[..., :n_L]
                u_part = vec[..., n_L:]

                # build L
                L = torch.eye(N, device=x.device).expand(B, C, N, N).clone()
                idx = torch.tril_indices(N, N, offset=-1)
                L[..., idx[0], idx[1]] = self.small_factor * torch.tanh(l_part)

                # build U
                U = torch.eye(N, device=x.device).expand(B, C, N, N).clone()
                offdiag_idx = torch.triu_indices(N, N, offset=1)
                U[..., offdiag_idx[0], offdiag_idx[1]] = self.small_factor * torch.tanh(u_part[..., :-N])
                diag_idx = torch.arange(N, device=x.device)
                U[..., diag_idx, diag_idx] = torch.exp(self.small_factor * u_part[..., -N:])  # ≈1

                mat = L @ U
                # scale = torch.exp(self.scale_val[h])
                # mat = scale * mat
                mats.append(mat)

            mat = torch.stack(mats, dim=2)  # (B, C, n_h, N, N)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return mat
#TODO: Add Transformer Layer to process wsfeat of the encoder

class TransformersLinearQuasi(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, mode = "pooled",init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec
        if mode == "flat":
            self.quasi_alpha = GLQuasiAlpha(encoder_weight_spec, in_channels, out_channels, init_type=init_type, scale_degree=scale_degree)
        elif mode == "pooled":
            self.quasi_alpha = GLQuasiAlphaPooled(encoder_weight_spec, in_channels, out_channels, init_type=init_type, scale_degree=scale_degree)
        elif mode == "pooledper":
            self.quasi_alpha = GLQuasiAlphaPooledPerLayer(encoder_weight_spec, in_channels, out_channels, init_type=init_type, scale_degree=scale_degree)
        elif mode == "flatper":
            self.quasi_alpha = GLQuasiAlphaPer(encoder_weight_spec, in_channels, out_channels, init_type=init_type, scale_degree=scale_degree)

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # -----------------------------------
            #            W_Q Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_Q",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_q],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_Q"), init_type=init_type)
            # -----------------------------------
            #            W_K Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_K",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_k],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_K"), init_type=init_type)
            # -----------------------------------
            #            W_V Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_V",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_v],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1],
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_V"), init_type=init_type)
            # -----------------------------------
            #            W_O Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_O_1", EinsumLayer(equation="bdhjk, ed -> behj",
                                        weight_shape=[self.e, self.d],
                                        input_shape=[-1, self.d, h, D_v, D],
                                        # fan_in_mask=[0, 1],
                                        fan_in_mask=[0, 1, 0, 0, 1],
                                        unsqueeze_dims=[-1]
                                        ))
            
            self.add_module(f"layer_{i}_W_O_2", EinsumLayer(equation="bdhjk, ed -> behjk",
                                        weight_shape=[self.e, self.d],
                                        input_shape=[-1, self.d, h, D_v, D],
                                        # fan_in_mask=[0, 1]
                                        fan_in_mask=[0, 1, 0, 0, 0]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_O_1"), getattr(self, f"layer_{i}_W_O_2"), init_type=init_type)

            # -----------------------------------
            #            W_A Terms
            # -----------------------------------
            # 1st Term
            self.add_module(f"layer_{i}_W_A_W_QK", EinsumLayer(equation="bdhpq, edpq -> be",
                                            weight_shape=[self.e, self.d, D, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            # fan_in_mask=[0, 1, 1, 1],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 2nd Term
            self.add_module(f"layer_{i}_W_A_W_VO_1", EinsumLayer(equation="bdhpq, edp -> be",
                                                weight_shape=[self.e, self.d, D],
                                                input_shape=[-1, self.d, h, D, D],
                                                fan_in_mask=[0, 1, 1, 1, 1],
                                                # fan_in_mask=[0, 1, 1],
                                                unsqueeze_dims=[-1, -1]))
            
            # 3rd Term
            self.add_module(f"layer_{i}_W_A_W_VO_2", EinsumLayer(equation="bdhpj, edp -> bej",
                                                weight_shape=[self.e, self.d, D],
                                                input_shape=[-1, self.d, h, D, D],
                                                fan_in_mask=[0, 1, 1, 1, 0],
                                                # fan_in_mask=[0, 1, 1],
                                                unsqueeze_dims=[-1]))

            # 4th Term
            self.add_module(f"layer_{i}_W_A_W_A_1", EinsumLayer(equation="bdpq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            # fan_in_mask=[0, 1],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 1],
                                            unsqueeze_dims=[-1, -1]))

            # 5th Term
            self.add_module(f"layer_{i}_W_A_W_A_2", EinsumLayer(equation="bdjq, ed -> bej",                                                               
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 0, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1]))
            
            # 6th Term
            self.add_module(f"layer_{i}_W_A_W_A_3", EinsumLayer(equation="bdpk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 0],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-2]))

            # 7th Term
            self.add_module(f"layer_{i}_W_A_W_A_4", EinsumLayer(equation="bdjk, ed -> bejk",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 0, 0],
                                            # fan_in_mask=[0, 1]
                                            ))
            
            # 8th Term
            self.add_module(f"layer_{i}_W_A_W_B_1", EinsumLayer(equation="bdpq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 9th Term
            self.add_module(f"layer_{i}_W_A_W_B_2", EinsumLayer(equation="bdkq, edq -> bek",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 0, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-2]))
            
            # 10th Term
            self.add_module(f"layer_{i}_W_A_b_A_1", EinsumLayer(equation="bdq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 11th Term
            self.add_module(f"layer_{i}_W_A_b_A_2", EinsumLayer(equation="bdk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 0],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-2]))

            # 12th Term
            self.add_module(f"layer_{i}_W_A_b_B", EinsumLayer(equation="bdq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1, -1]))

            # 13th Term
            self.add_module(f"layer_{i}_W_A_bias", EinsumLayer(equation="e -> e",
                                        weight_shape=[self.e],
                                        input_shape=[self.e],
                                        fan_in_mask=[0],
                                        # fan_in_mask=[0],
                                        unsqueeze_dims=[0, -1, -1]))
            
            set_init_einsum_(
                            getattr(self, f"layer_{i}_W_A_b_A_2"),
                            getattr(self, f"layer_{i}_W_A_b_B"),
                            getattr(self, f"layer_{i}_W_A_bias"),
                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_W_A_W_QK"),
                            getattr(self, f"layer_{i}_W_A_W_VO_1"),
                            getattr(self, f"layer_{i}_W_A_W_VO_2"),
init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_W_A_b_A_1"),
                            getattr(self, f"layer_{i}_W_A_W_B_1"),
                            getattr(self, f"layer_{i}_W_A_W_B_2"),
                            getattr(self, f"layer_{i}_W_A_W_A_1"),
                            getattr(self, f"layer_{i}_W_A_W_A_2"),
                            getattr(self, f"layer_{i}_W_A_W_A_3"),
                            getattr(self, f"layer_{i}_W_A_W_A_4"),

init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            b_A Terms
            # -----------------------------------
            # 1st Term
            self.add_module(f"layer_{i}_b_A_QK", EinsumLayer(equation="bdhpq, edpq -> be",
                                            weight_shape=[self.e, self.d, D, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1, 1],
                                            unsqueeze_dims=[-1]))
            
            # 2nd Term
            self.add_module(f"layer_{i}_b_A_VO", EinsumLayer(equation="bdhpq, edp -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))
            
            # 3rd Term
            self.add_module(f"layer_{i}_b_A_W_A_1",  EinsumLayer(equation="bdpq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1]))

            # 4th Term
            self.add_module(f"layer_{i}_b_A_W_A_2", EinsumLayer(equation="bdpk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 0],
                                            # fan_in_mask=[0, 1]
                                            ))

            # 5th Term
            self.add_module(f"layer_{i}_b_A_W_B_1", EinsumLayer(equation="bdpq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 1 ,1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))

            # 6th Term
            self.add_module(f"layer_{i}_b_A_W_B_2", EinsumLayer(equation="bdkq, edq -> bek",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 0, 1]
                                            # fan_in_mask=[0, 1, 1]
                                            ))

            # 7th Term
            self.add_module(f"layer_{i}_b_A_b_A_1",  EinsumLayer(equation="bdq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]
                                            # fan_in_mask=[0, 1, 1, 0]
                                            ))

            # 8th Term
            self.add_module(f"layer_{i}_b_A_b_A_2", EinsumLayer(equation="bdk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 0],
                                            # fan_in_mask=[0, 1, 0]
                                            ))

            # 9th Term
            self.add_module(f"layer_{i}_b_A_b_B",  EinsumLayer(equation="bdq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))

            # 10th Term
            self.add_module(f"layer_{i}_b_A_bias", EinsumLayer(equation="e -> e",
                                            weight_shape=[self.e],
                                            input_shape=[self.e],
                                            fan_in_mask=[0],
                                            # fan_in_mask=[0],
                                            unsqueeze_dims=[0, -1]))
            
            set_init_einsum_(
                
                            getattr(self, f"layer_{i}_b_A_b_A_2"),
                            getattr(self, f"layer_{i}_b_A_b_B"),
                            getattr(self, f"layer_{i}_b_A_bias"),

                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_b_A_QK"),
                            getattr(self, f"layer_{i}_b_A_VO"),
                            init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_b_A_W_B_1"),
                            getattr(self, f"layer_{i}_b_A_W_B_2"), getattr(self, f"layer_{i}_b_A_W_A_1"),
                            getattr(self, f"layer_{i}_b_A_W_A_2"),getattr(self, f"layer_{i}_b_A_b_A_1"),  #bAbA1
                            init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            W_B Terms
            # -----------------------------------
            
            # 1st Term
            self.add_module(f"layer_{i}_W_B_QK_1", EinsumLayer(
                equation="bdhpq, edkpq -> bek",
                weight_shape=[self.e, self.d, D, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1, 1],
                unsqueeze_dims=[-2]
            ))

            # 2nd Term
            self.add_module(f"layer_{i}_W_B_VO", EinsumLayer(
                equation="bdhpq, edkp -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 3rd Term
            self.add_module(f"layer_{i}_W_B_W_A_1", EinsumLayer(
                equation="bdpq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 1],
                # fan_in_mask=[0, 1, 0],
                unsqueeze_dims=[-2]
            ))

            # 4th Term
            self.add_module(f"layer_{i}_W_B_W_A_2", EinsumLayer(
                equation="bdpj, edk -> bejk",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 0],
                # fan_in_mask=[0, 1, 0]
            ))

            # 5th Term
            self.add_module(f"layer_{i}_W_B_W_B_1", EinsumLayer(
                equation="bdpq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 6th Term
            self.add_module(f"layer_{i}_W_B_W_B_2",  EinsumLayer(
                equation="bdjq, edkq -> bejk",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 0, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 7th Term
            self.add_module(f"layer_{i}_W_B_b_A_1", EinsumLayer(
                equation="bdq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 1],
                # fan_in_mask=[0, 1, 0],
                unsqueeze_dims=[-2]
            ))

            # 8th Term
            self.add_module(f"layer_{i}_W_B_b_A_2", EinsumLayer(
                equation="bdj, edk -> bejk",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 0]
                # fan_in_mask=[0, 1, 0]
            ))

            # 9th Term
            self.add_module(f"layer_{i}_W_B_b_B", EinsumLayer(
                equation="bdq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D],
                fan_in_mask=[0, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 10th Term (Bias Term)
            self.add_module(f"layer_{i}_W_B_bias", EinsumLayer(
                equation="ek -> ek",
                weight_shape=[self.e, D],
                input_shape=[self.e, D],
                fan_in_mask=[0, 0],
                # fan_in_mask=[0, 0],
                unsqueeze_dims=[0, -2]
            ))
            
            set_init_einsum_(
                            getattr(self, f"layer_{i}_W_B_b_A_2"),
                            getattr(self, f"layer_{i}_W_B_b_B"),
                            getattr(self, f"layer_{i}_W_B_bias"),
                            

                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_W_B_QK_1"),
                            getattr(self, f"layer_{i}_W_B_VO"), init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_W_B_W_B_1"),
                            getattr(self, f"layer_{i}_W_B_b_A_1"),getattr(self, f"layer_{i}_W_B_W_A_1"),
                            getattr(self, f"layer_{i}_W_B_W_A_2"), getattr(self, f"layer_{i}_W_B_W_B_2"), init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            b_B Terms
            # -----------------------------------                                   
            # 1st Term
            self.add_module(f"layer_{i}_b_B_QK", EinsumLayer(
                equation="bdhpq, edkpq -> bek",
                weight_shape=[self.e, self.d, D, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1, 1]
            ))

            # 2nd Term
            self.add_module(f"layer_{i}_b_B_VO", EinsumLayer(
                equation="bdhpq, edkp -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 3rd Term
            self.add_module(f"layer_{i}_b_B_W_A", EinsumLayer(
                equation="bdpq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 1]
                # fan_in_mask=[0, 1, 0]
            ))

            # 4th Term
            self.add_module(f"layer_{i}_b_B_W_B", EinsumLayer(
                equation="bdpq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 5th Term
            self.add_module(f"layer_{i}_b_B_b_A", EinsumLayer(
                equation="bdq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 1]
                # fan_in_mask=[0, 1, 0]
            ))

            # 6th Term
            self.add_module(f"layer_{i}_b_B_b_B", EinsumLayer(
                equation="bdq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D],
                fan_in_mask=[0, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 7th Term: Bias Term as EinsumLayer
            self.add_module(f"layer_{i}_b_B_bias", EinsumLayer(
                equation="ed -> ed",
                weight_shape=[self.e, D],
                input_shape=[self.e, D],
                fan_in_mask=[0, 0],
                # fan_in_mask=[0, 0],
                unsqueeze_dims=[0]
            ))
            set_init_einsum_(                        
                        getattr(self, f"layer_{i}_b_B_b_A"),
                        getattr(self, f"layer_{i}_b_B_b_B"),
                        getattr(self, f"layer_{i}_b_B_bias"),
                        
            init_type=init_type)
            set_init_einsum_(
                        getattr(self, f"layer_{i}_b_B_QK"),
                        getattr(self, f"layer_{i}_b_B_VO"),init_type=init_type,scale_degree=scale_degree)
            set_init_einsum_(
                        getattr(self, f"layer_{i}_b_B_W_B"),
                        getattr(self, f"layer_{i}_b_B_b_A"), 
                        getattr(self, f"layer_{i}_b_B_W_B"),getattr(self, f"layer_{i}_b_B_W_A"),init_type=init_type,scale_degree=scale_degree)


    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        all_M, all_N = self.quasi_alpha(wsfeat)
        out_dict = {
            "W_q": [], "W_k": [], "W_v": [], "W_o": [],
            "W_A": [], "W_B": [], "b_A": [], "b_B": []
        }
        
        L = len(wsfeat)  # Number of layers
    
        # Loop over each layer's weights and biases
        for i in range(L):
            M, N = all_M[i], all_N[i]
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]

            # Compute intermediate products using einsum equations
            WW_qk = torch.einsum('bdhpk, bdhqk -> bdhpq', W_q, W_k)
            WW_vo = torch.einsum('bdhpk, bdhkq -> bdhpq', W_v, W_o)

            # Apply your EinsumLayers for W_q, W_k, W_v, and W_o
            layer_W_q = getattr(self, f"layer_{i}_W_Q")(W_q)
            layer_W_k = getattr(self, f"layer_{i}_W_K")(W_k)
            layer_W_v = getattr(self, f"layer_{i}_W_V")(W_v)
            layer_W_o_1 = getattr(self, f"layer_{i}_W_O_1")(W_o)
            layer_W_o_2 = getattr(self, f"layer_{i}_W_O_2")(W_o)

            # Apply your EinsumLayers for W_A
            layer_W_A_W_QK = getattr(self, f"layer_{i}_W_A_W_QK")(WW_qk)
            layer_W_A_W_VO_1 = getattr(self, f"layer_{i}_W_A_W_VO_1")(WW_vo)
            layer_W_A_W_VO_2 = getattr(self, f"layer_{i}_W_A_W_VO_2")(WW_vo)
            layer_W_A_W_A_1 = getattr(self, f"layer_{i}_W_A_W_A_1")(W_A)
            layer_W_A_W_A_2 = getattr(self, f"layer_{i}_W_A_W_A_2")(W_A)
            layer_W_A_W_A_3 = getattr(self, f"layer_{i}_W_A_W_A_3")(W_A)
            layer_W_A_W_A_4 = getattr(self, f"layer_{i}_W_A_W_A_4")(W_A)
            layer_W_A_W_B_1 = getattr(self, f"layer_{i}_W_A_W_B_1")(W_B)
            layer_W_A_W_B_2 = getattr(self, f"layer_{i}_W_A_W_B_2")(W_B)
            layer_W_A_b_A_1 = getattr(self, f"layer_{i}_W_A_b_A_1")(b_A)
            layer_W_A_b_A_2 = getattr(self, f"layer_{i}_W_A_b_A_2")(b_A)
            layer_W_A_b_B = getattr(self, f"layer_{i}_W_A_b_B")(b_B)
            layer_W_A_bias = getattr(self, f"layer_{i}_W_A_bias")()

            # Apply your EinsumLayers for b_A
            layer_b_A_QK = getattr(self, f"layer_{i}_b_A_QK")(WW_qk)
            layer_b_A_VO = getattr(self, f"layer_{i}_b_A_VO")(WW_vo)
            layer_b_A_W_A_1 = getattr(self, f"layer_{i}_b_A_W_A_1")(W_A)
            layer_b_A_W_A_2 = getattr(self, f"layer_{i}_b_A_W_A_2")(W_A)
            layer_b_A_W_B_1 = getattr(self, f"layer_{i}_b_A_W_B_1")(W_B)
            layer_b_A_W_B_2 = getattr(self, f"layer_{i}_b_A_W_B_2")(W_B)
            layer_b_A_b_A_1 = getattr(self, f"layer_{i}_b_A_b_A_1")(b_A)
            layer_b_A_b_A_2 = getattr(self, f"layer_{i}_b_A_b_A_2")(b_A)
            layer_b_A_b_B = getattr(self, f"layer_{i}_b_A_b_B")(b_B)
            layer_b_A_bias = getattr(self, f"layer_{i}_b_A_bias")()

            # Apply your EinsumLayers for W_B
            layer_W_B_QK_1 = getattr(self, f"layer_{i}_W_B_QK_1")(WW_qk)
            layer_W_B_VO = getattr(self, f"layer_{i}_W_B_VO")(WW_vo)
            layer_W_B_W_A_1 = getattr(self, f"layer_{i}_W_B_W_A_1")(W_A)
            layer_W_B_W_A_2 = getattr(self, f"layer_{i}_W_B_W_A_2")(W_A)
            layer_W_B_W_B_1 = getattr(self, f"layer_{i}_W_B_W_B_1")(W_B)
            layer_W_B_W_B_2 = getattr(self, f"layer_{i}_W_B_W_B_2")(W_B)
            layer_W_B_b_A_1 = getattr(self, f"layer_{i}_W_B_b_A_1")(b_A)
            layer_W_B_b_A_2 = getattr(self, f"layer_{i}_W_B_b_A_2")(b_A)
            layer_W_B_b_B = getattr(self, f"layer_{i}_W_B_b_B")(b_B)
            layer_W_B_bias = getattr(self, f"layer_{i}_W_B_bias")()

            # Apply your EinsumLayers for b_B
            layer_b_B_QK = getattr(self, f"layer_{i}_b_B_QK")(WW_qk)
            layer_b_B_VO = getattr(self, f"layer_{i}_b_B_VO")(WW_vo)
            layer_b_B_W_A = getattr(self, f"layer_{i}_b_B_W_A")(W_A)
            layer_b_B_W_B = getattr(self, f"layer_{i}_b_B_W_B")(W_B)
            layer_b_B_b_A = getattr(self, f"layer_{i}_b_B_b_A")(b_A)
            layer_b_B_b_B = getattr(self, f"layer_{i}_b_B_b_B")(b_B)
            layer_b_B_bias = getattr(self, f"layer_{i}_b_B_bias")()
            # print(layer_W_q.shape)
            # print(M.shape)
            out_dict["W_q"].append(layer_W_q @ M.transpose(-1,-2))
            out_dict["W_k"].append(layer_W_k @ torch.inverse(M))
            out_dict["W_v"].append(layer_W_v @ N)
            out_dict["W_o"].append(N @ (layer_W_o_1 + layer_W_o_2))
            
            out_dict["W_A"].append(
                (layer_W_A_W_QK + layer_W_A_W_VO_1 + layer_W_A_W_VO_2 +
                layer_W_A_W_A_1 + layer_W_A_W_A_2 + layer_W_A_W_A_3 +
                layer_W_A_W_A_4 + layer_W_A_W_B_1 + layer_W_A_W_B_2 +
                layer_W_A_b_A_1 + layer_W_A_b_A_2 + layer_W_A_b_B +
                layer_W_A_bias)
            )
            out_dict["W_B"].append(
                (layer_W_B_QK_1 + layer_W_B_VO + layer_W_B_W_A_1 +
                layer_W_B_W_A_2 + layer_W_B_W_B_1 + layer_W_B_W_B_2 +
                layer_W_B_b_A_1 + layer_W_B_b_A_2 + layer_W_B_b_B +
                layer_W_B_bias)
            )

            out_dict["b_A"].append(
                (layer_b_A_QK + layer_b_A_VO + layer_b_A_W_A_1 +
                layer_b_A_W_A_2 + layer_b_A_W_B_1 + layer_b_A_W_B_2 +
                layer_b_A_b_A_1 + layer_b_A_b_A_2 + layer_b_A_b_B +
                layer_b_A_bias)
            )

            out_dict["b_B"].append(
                (layer_b_B_QK + layer_b_B_VO + layer_b_B_W_A +
                layer_b_B_W_B + layer_b_B_b_A + layer_b_B_b_B +
                layer_b_B_bias)
            )

        return AttentionWeightSpaceFeatures(**out_dict)
