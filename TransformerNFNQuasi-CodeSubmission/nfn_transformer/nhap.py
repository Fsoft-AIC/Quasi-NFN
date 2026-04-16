import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Alternative 1: Hierarchical Feature Processing with Skip Connections
class GLQuasiAlphaHierarchical(nn.Module):
    def __init__(self, encoder_weight_spec, in_channels, out_channels, 
                 init_type="pytorch_default", scale_degree=2, base_dim=16):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec
        D, D_q, D_k, D_v, D_a, h = encoder_weight_spec.get_all_dims()
        self.L = len(encoder_weight_spec)
        
        # Hierarchical processing: compress features first, then expand
        input_dim = 7 * 8 * self.L
        self.feature_compressor = nn.Sequential(
            nn.Linear(input_dim, base_dim * 4),
            nn.LayerNorm(base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, base_dim),
            nn.LayerNorm(base_dim)
        )
        
        # Layer-specific adapters with residual connections
        for i in range(self.L):
            # Smaller networks that work on compressed features
            self.add_module(f"layer_{i}_M_adapter", 
                nn.Sequential(
                    nn.Linear(base_dim, base_dim // 2),
                    nn.GELU(),
                    nn.Linear(base_dim // 2, base_dim)
                ))
            
            self.add_module(f"layer_{i}_M",
                SimpleMLPGL_v2(input_dim=base_dim, output_dim=D_k, n_h=h, 
                             hidden_dims=[base_dim], mode="lu", use_residual=True))
            
            self.add_module(f"layer_{i}_N_adapter",
                nn.Sequential(
                    nn.Linear(base_dim, base_dim // 2),
                    nn.GELU(),
                    nn.Linear(base_dim // 2, base_dim)
                ))
                
            self.add_module(f"layer_{i}_N",
                SimpleMLPGL_v2(input_dim=base_dim, output_dim=D_v, n_h=h,
                             hidden_dims=[base_dim], mode="lu", use_residual=True))

    def forward(self, wsfeat):
        # Extract features (same as original)
        all_M, all_N = [], []
        L = len(wsfeat)
        out = []
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)
        
        for i in range(L):
            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)
                feats.append(torch.stack([mean, var, *q], dim=-1))
            out.append(torch.cat(feats, dim=-1))
        
        out = torch.cat(out, dim=-1)
        
        # Compress features once
        compressed_feat = self.feature_compressor(out)
        
        # Generate M and N with layer-specific adaptation
        for i in range(L):
            # Layer-specific feature adaptation
            adapted_M = getattr(self, f"layer_{i}_M_adapter")(compressed_feat)
            adapted_N = getattr(self, f"layer_{i}_N_adapter")(compressed_feat)
            
            # Add residual connection
            adapted_M = adapted_M + compressed_feat
            adapted_N = adapted_N + compressed_feat
            
            M = getattr(self, f"layer_{i}_M")(adapted_M)
            N = getattr(self, f"layer_{i}_N")(adapted_N)
            
            all_M.append(M)
            all_N.append(N)
            
        return all_M, all_N


# Alternative 2: Shared Base with Layer-specific Heads
class GLQuasiAlphaShared(nn.Module):
    def __init__(self, encoder_weight_spec, in_channels, out_channels,
                 init_type="pytorch_default", scale_degree=2, base_dim=32):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec
        D, D_q, D_k, D_v, D_a, h = encoder_weight_spec.get_all_dims()
        self.L = len(encoder_weight_spec)
        
        input_dim = 7 * 8 * self.L
        
        # Shared base encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim * 2),
            nn.LayerNorm(base_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim * 2, base_dim),
            nn.LayerNorm(base_dim)
        )
        
        # Layer embedding for layer-specific information
        self.layer_embedding = nn.Embedding(self.L, base_dim // 4)
        
        # Shared M and N generators with layer conditioning
        self.M_generator = SimpleMLPGL_v2(
            input_dim=base_dim + base_dim // 4, output_dim=D_k, n_h=h,
            hidden_dims=[base_dim // 2], mode="lu", use_layer_norm=True
        )
        
        self.N_generator = SimpleMLPGL_v2(
            input_dim=base_dim + base_dim // 4, output_dim=D_v, n_h=h,
            hidden_dims=[base_dim // 2], mode="lu", use_layer_norm=True
        )

    def forward(self, wsfeat):
        # Feature extraction (same as original)
        all_M, all_N = [], []
        L = len(wsfeat)
        out = []
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)
        
        for i in range(L):
            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)
                feats.append(torch.stack([mean, var, *q], dim=-1))
            out.append(torch.cat(feats, dim=-1))
        
        out = torch.cat(out, dim=-1)
        
        # Shared encoding
        shared_feat = self.shared_encoder(out)
        
        # Generate layer-specific M and N
        for i in range(L):
            layer_emb = self.layer_embedding(torch.tensor(i, device=out.device)).unsqueeze(0).expand(out.size(0), -1)
            conditioned_feat = torch.cat([shared_feat, layer_emb], dim=-1)
            
            M = self.M_generator(conditioned_feat)
            N = self.N_generator(conditioned_feat)
            
            all_M.append(M)
            all_N.append(N)
            
        return all_M, all_N


# Alternative 3: Low-rank Factorization Approach
class GLQuasiAlphaLowRank(nn.Module):
    def __init__(self, encoder_weight_spec, in_channels, out_channels,
                 init_type="pytorch_default", scale_degree=2, rank=8):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec
        D, D_q, D_k, D_v, D_a, h = encoder_weight_spec.get_all_dims()
        self.L = len(encoder_weight_spec)
        self.rank = rank
        
        input_dim = 7 * 8 * self.L
        
        # Feature processor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        
        # Low-rank factorization: instead of generating full matrices,
        # generate low-rank factors
        for i in range(self.L):
            # M = U_M @ V_M, where U_M is D_k x rank, V_M is rank x D_k
            self.add_module(f"layer_{i}_M_U", 
                nn.Linear(32, D_k * rank * h))
            self.add_module(f"layer_{i}_M_V", 
                nn.Linear(32, rank * D_k * h))
            
            self.add_module(f"layer_{i}_N_U", 
                nn.Linear(32, D_v * rank * h))
            self.add_module(f"layer_{i}_N_V", 
                nn.Linear(32, rank * D_v * h))

    def forward(self, wsfeat):
        # Feature extraction (same as original)
        all_M, all_N = [], []
        L = len(wsfeat)
        out = []
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)
        
        for i in range(L):
            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)
                feats.append(torch.stack([mean, var, *q], dim=-1))
            out.append(torch.cat(feats, dim=-1))
        
        out = torch.cat(out, dim=-1)
        
        # Process features
        feat = self.feature_net(out)
        B = feat.shape[0]
        
        # Generate low-rank factors and reconstruct matrices
        D, D_q, D_k, D_v, D_a, h = self.encoder_weight_spec.get_all_dims()
        
        for i in range(L):
            # Generate M via low-rank factorization
            U_M = getattr(self, f"layer_{i}_M_U")(feat).view(B, h, D_k, self.rank)
            V_M = getattr(self, f"layer_{i}_M_V")(feat).view(B, h, self.rank, D_k)
            M = torch.matmul(U_M, V_M)
            
            # Add identity for stability
            eye = torch.eye(D_k, device=feat.device).view(1, 1, D_k, D_k)
            M = M + 0.1 * eye
            
            # Generate N via low-rank factorization  
            U_N = getattr(self, f"layer_{i}_N_U")(feat).view(B, h, D_v, self.rank)
            V_N = getattr(self, f"layer_{i}_N_V")(feat).view(B, h, self.rank, D_v)
            N = torch.matmul(U_N, V_N)
            
            # Add identity for stability
            eye = torch.eye(D_v, device=feat.device).view(1, 1, D_v, D_v)
            N = N + 0.1 * eye
            
            all_M.append(M)
            all_N.append(N)
            
        return all_M, all_N


# Improved SimpleMLPGL with better conditioning and regularization
class SimpleMLPGL_v2(nn.Module):
    def __init__(self, input_dim, output_dim, n_h, hidden_dims=[16],
                 activation=nn.GELU(), init_type="xavier", mode="lu",
                 use_residual=False, use_layer_norm=True, dropout_rate=0.1):
        super().__init__()
        self.mode = mode
        self.out_dim = output_dim
        self.n_h = n_h
        self.init_type = init_type
        self.use_residual = use_residual
        self.compute_loss = False
        
        if mode == "lu":
            N = output_dim
            n_L = N * (N - 1) // 2
            n_U = N * (N + 1) // 2
            final_out = (n_L + n_U) * n_h
        else:
            final_out = output_dim * output_dim * n_h
        
        dims = [input_dim] + hidden_dims + [final_out]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:  # not the last layer
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.net = nn.Sequential(*layers)
        self.current_reg_loss = 0
        
        # Residual connection if input and output dims match
        if use_residual and input_dim == final_out:
            self.residual_proj = nn.Identity()
        elif use_residual:
            self.residual_proj = nn.Linear(input_dim, final_out)
        else:
            self.residual_proj = None
            
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

    def forward(self, x):
        B = x.shape[0]
        raw = self.net(x)
        
        # Add residual connection if available
        if self.residual_proj is not None:
            raw = raw + self.residual_proj(x)
        
        if self.mode == "lu":
            N = self.out_dim
            n_L = N * (N - 1) // 2
            n_U = N * (N + 1) // 2
            total = n_L + n_U
            raw = raw.view(B, self.n_h, total)
            
            mats = []
            for h in range(self.n_h):
                vec = raw[:, h, :]
                l_part = vec[:, :n_L]
                u_part = vec[:, n_L:]
                
                # Build L (lower triangular with 1s on diagonal)
                L = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
                idx = torch.tril_indices(N, N, offset=-1)
                L[:, idx[0], idx[1]] = l_part
                
                # Build U (upper triangular)
                U = torch.zeros(B, N, N, device=x.device)
                idx = torch.triu_indices(N, N)
                U[:, idx[0], idx[1]] = u_part
                
                # Ensure positive diagonal for U with better conditioning
                diag_idx = torch.arange(N, device=x.device)
                U[:, diag_idx, diag_idx] = F.softplus(U[:, diag_idx, diag_idx]) + 1e-2
                
                mats.append(L @ U)
            
            mat = torch.stack(mats, dim=1)
            
        return mat


# Alternative 4: Attention-based Feature Processing
class GLQuasiAlphaAttention(nn.Module):
    def __init__(self, encoder_weight_spec, in_channels, out_channels,
                 init_type="pytorch_default", scale_degree=2, embed_dim=64):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec
        D, D_q, D_k, D_v, D_a, h = encoder_weight_spec.get_all_dims()
        self.L = len(encoder_weight_spec)
        self.embed_dim = embed_dim
        
        # Feature embedding
        input_dim = 7 * 8
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        
        # Multi-head attention to process layer interactions
        self.layer_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Layer positional encoding
        self.layer_pos_encoding = nn.Parameter(torch.randn(self.L, embed_dim) * 0.02)
        
        # Output projections
        self.M_proj = SimpleMLPGL_v2(
            input_dim=embed_dim, output_dim=D_k, n_h=h,
            hidden_dims=[embed_dim // 2], mode="lu"
        )
        self.N_proj = SimpleMLPGL_v2(
            input_dim=embed_dim, output_dim=D_v, n_h=h,
            hidden_dims=[embed_dim // 2], mode="lu"
        )

    def forward(self, wsfeat):
        all_M, all_N = [], []
        L = len(wsfeat)
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=wsfeat[0][0].device)
        
        # Process each layer's features separately
        layer_features = []
        for i in range(L):
            terms = [torch.flatten(x, start_dim=2) for x in wsfeat[i]]
            feats = []
            for t in terms:
                mean, var = t.mean(-1), t.var(-1)
                q = torch.quantile(t, quantiles, dim=-1)
                feats.append(torch.stack([mean, var, *q], dim=-1))
            layer_feat = torch.cat(feats, dim=-1)  # (B, C, 7*8)
            layer_features.append(layer_feat)
        
        # Stack and embed: (B, L, C, embed_dim)
        layer_stack = torch.stack(layer_features, dim=1)  # (B, L, C, 7*8)
        B, L, C, _ = layer_stack.shape
        
        # Embed features
        embedded = self.feature_embedding(layer_stack)  # (B, L, C, embed_dim)
        
        # Add positional encoding
        embedded = embedded + self.layer_pos_encoding.unsqueeze(0).unsqueeze(2)
        
        # Reshape for attention: (B*C, L, embed_dim)
        embedded_flat = embedded.view(B * C, L, self.embed_dim)
        
        # Apply attention across layers
        attended, _ = self.layer_attention(embedded_flat, embedded_flat, embedded_flat)
        
        # Reshape back: (B, L, C, embed_dim)
        attended = attended.view(B, L, C, self.embed_dim)
        
        # Generate M and N for each layer
        for i in range(L):
            layer_feat = attended[:, i]  # (B, C, embed_dim)
            # Average across channels for matrix generation
            pooled_feat = layer_feat.mean(dim=1)  # (B, embed_dim)
            
            M = self.M_proj(pooled_feat)
            N = self.N_proj(pooled_feat)
            
            all_M.append(M)
            all_N.append(N)
        
        return all_M, all_N