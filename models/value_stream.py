"""
Value Stream Components: ValueTransformer, Discriminator, TokenGenerator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueTransformer(nn.Module):
    """
    Context-aware Value Transformer
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        value_dim: int = 128,
        n_self_attn_layers: int = 2, 
        n_heads: int = 4,
        dropout: float = 0.1,
        use_attention_pooling: bool = False, 
        use_transformer_aggregate: bool = False, 
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.value_dim = value_dim
        self.n_self_attn_layers = n_self_attn_layers
        self.use_attention_pooling = use_attention_pooling
        self.use_transformer_aggregate = use_transformer_aggregate
        
        # ===== Stage 1 Components (unconditional pre-training) =====
        # Response Proj: Project hidden states to value space
        self.response_proj = nn.Sequential(
            nn.Linear(hidden_dim, value_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=value_dim,
                nhead=n_heads,
                dim_feedforward=value_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(n_self_attn_layers)
        ])
        
        # ===== Stage 2 Components (conditional fine-tuning) =====
        # Prompt Proj: Project prompt hidden states to value space
        self.prompt_proj = nn.Sequential(
            nn.Linear(hidden_dim, value_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=value_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(value_dim)
        
        # Cross-Attention scale (learnable, control the contribution of Cross-Attention)
        self.cross_scale = nn.Parameter(torch.ones(1) * 1.0)
            
        self._init_stage2_components()
        
        # Attention pooling query (for use_attention_pooling=True)
        if use_attention_pooling:
            self.pooling_query = nn.Parameter(torch.randn(1, value_dim))
            nn.init.xavier_uniform_(self.pooling_query, gain=0.1)
        
        # Aggregate network
        if use_transformer_aggregate:
            # Use Transformer Encoder for stronger expressiveness
            self.aggregate = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=value_dim,
                    nhead=n_heads,
                    dim_feedforward=value_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=2
            )
        else:
            # Original simple MLP
            self.aggregate = nn.Sequential(
                nn.Linear(value_dim, value_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(value_dim),
            )
    
    def _init_stage2_components(self):
        for module in self.prompt_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in [self.cross_attn]:
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)  
            if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight, gain=1.0) 
        
        nn.init.ones_(self.cross_norm.weight)
        nn.init.zeros_(self.cross_norm.bias)
        
        nn.init.constant_(self.cross_scale, 3.0)
    
    def forward_unconditional(
        self,
        hidden_states: torch.Tensor,      # [batch, seq_len, hidden_dim]
        attention_mask: torch.Tensor = None,  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Stage 1: Unconditional pre-training 
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] from base model
            attention_mask: [batch, seq_len] (1 = not masked, 0 = masked)
        
        Returns:
            value_repr: [batch, value_dim]
        """
        x = self.response_proj(hidden_states)  # [batch, seq_len, value_dim]
        
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.self_attn_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        # x: [batch, seq_len, value_dim]

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  
            seq_lengths = torch.clamp(seq_lengths, min=0)  
            batch_indices = torch.arange(x.size(0), device=x.device)
            x = x[batch_indices, seq_lengths]  # [batch, value_dim]
        else:
            x = x[:, -1, :]  # [batch, value_dim]

        value_repr = self.aggregate(x)  # [batch, value_dim]        
        return value_repr
    
    def forward_conditional(
        self,
        prompt_hidden: torch.Tensor,      # [batch, prompt_len, hidden_dim]
        response_hidden: torch.Tensor,     # [batch, response_len, hidden_dim]
        prompt_mask: torch.Tensor = None,  # [batch, prompt_len]
        response_mask: torch.Tensor = None, # [batch, response_len]
    ) -> torch.Tensor:
        """
        Stage 2: Conditional fine-tuning 
        
        Args:
            prompt_hidden: [batch, prompt_len, hidden_dim] prompt hidden states
            response_hidden: [batch, response_len, hidden_dim] response hidden states
            prompt_mask: [batch, prompt_len] (1 = not masked, 0 = masked)
            response_mask: [batch, response_len] (1 = not masked, 0 = masked)
        
        Returns:
            value_repr: [batch, value_dim]
        """
        prompt_value = self.prompt_proj(prompt_hidden)  # [batch, prompt_len, value_dim]
        response_value = self.response_proj(response_hidden)  # [batch, response_len, value_dim]
        
        prompt_value = F.layer_norm(prompt_value, prompt_value.shape[-1:])
        response_value = F.layer_norm(response_value, response_value.shape[-1:])

        attn_output, _ = self.cross_attn(
            query=response_value,
            key=prompt_value,
            value=prompt_value,
            key_padding_mask=(prompt_mask == 0) if prompt_mask is not None else None,
        )
        z_cross = self.cross_norm(attn_output)  # [batch, response_len, value_dim]
        

        z_conditional = response_value + self.cross_scale * z_cross
        x = z_conditional
        key_padding_mask = (response_mask == 0) if response_mask is not None else None
        for layer in self.self_attn_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        # x: [batch, response_len, value_dim]
        
        if response_mask is not None:
            seq_lengths = response_mask.sum(dim=1) - 1  
            seq_lengths = torch.clamp(seq_lengths, min=0)  
            batch_indices = torch.arange(x.size(0), device=x.device)
            x = x[batch_indices, seq_lengths]  # [batch, value_dim]
        else:
            x = x[:, -1, :]  # [batch, value_dim]
        
        value_repr = self.aggregate(x)  # [batch, value_dim]        
        return value_repr
    
    def forward_stage3(
        self,
        prompt_hidden: torch.Tensor,      # [batch, prompt_len, hidden_dim]
        prompt_mask: torch.Tensor = None,  # [batch, prompt_len]
    ) -> torch.Tensor:
        """
        Stage 3: Extract Prompt Value (for generating intervention tokens)
        
        Args:
            prompt_hidden: [batch, prompt_len, hidden_dim] prompt hidden states
            prompt_mask: [batch, prompt_len] (1 = not masked, 0 = masked)
        
        Returns:
            prompt_value: [batch, value_dim]
        """
        prompt_value = self.prompt_proj(prompt_hidden)  # [batch, prompt_len, value_dim]

        if torch.isnan(prompt_value).any() or torch.isinf(prompt_value).any():
            prompt_value = torch.clamp(prompt_value, min=-1e6, max=1e6)
            prompt_value = torch.nan_to_num(prompt_value, nan=0.0, posinf=1e6, neginf=-1e6)
        
        key_padding_mask = (prompt_mask == 0) if prompt_mask is not None else None
        for layer in self.self_attn_layers:
            prompt_value = layer(prompt_value, src_key_padding_mask=key_padding_mask)
            if torch.isnan(prompt_value).any() or torch.isinf(prompt_value).any():
                prompt_value = torch.clamp(prompt_value, min=-1e6, max=1e6)
                prompt_value = torch.nan_to_num(prompt_value, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Aggregate sequence to single vector
        if self.use_attention_pooling:
            # Attention pooling: learnable weighted aggregation
            # Compute attention scores: [batch, seq_len, 1]
            attn_scores = torch.matmul(prompt_value, self.pooling_query.T) / (self.value_dim ** 0.5)  # [batch, seq_len, 1]
            
            # Apply mask if available
            if prompt_mask is not None:
                mask = (prompt_mask == 0).unsqueeze(-1)  # [batch, seq_len, 1]
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            
            # Softmax to get attention weights
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            
            # Weighted sum
            x = (prompt_value * attn_weights).sum(dim=1)  # [batch, value_dim]
        else:
            if prompt_mask is not None:
                seq_lengths = prompt_mask.sum(dim=1) - 1  
                seq_lengths = torch.clamp(seq_lengths, min=0)  
                batch_indices = torch.arange(prompt_value.size(0), device=prompt_value.device)
                x = prompt_value[batch_indices, seq_lengths]  # [batch, value_dim]
            else:
                x = prompt_value[:, -1, :]  # [batch, value_dim]
        
        # Apply aggregate network
        if self.use_transformer_aggregate:
            # Transformer encoder expects [batch, seq_len, value_dim]
            x = x.unsqueeze(1)  # [batch, 1, value_dim]
            x = self.aggregate(x)  # [batch, 1, value_dim]
            prompt_value_repr = x[:, 0, :]  # [batch, value_dim]
        else:
            prompt_value_repr = self.aggregate(x)  # [batch, value_dim]

        if torch.isnan(prompt_value_repr).any() or torch.isinf(prompt_value_repr).any():
            prompt_value_repr = torch.clamp(prompt_value_repr, min=-1e6, max=1e6)
            prompt_value_repr = torch.nan_to_num(prompt_value_repr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return prompt_value_repr
    
    def load_stage1_weights(self, state_dict: dict):
        """
        Load stage 1 weights (for stage 2 fine-tuning)

        """
        filtered_dict = {}
        current_state_dict = self.state_dict()
        
        for key, value in state_dict.items():
            new_key = None
            if key.startswith('prompt_proj') or key.startswith('cross_attn') or key.startswith('cross_norm'):
                continue
            if key.startswith('response_proj') or key.startswith('self_attn_layers') or key.startswith('aggregate'):
                new_key = key
            elif key.startswith('input_proj'):
                new_key = key.replace('input_proj', 'response_proj.0')
            elif key.startswith('transformer.layers'):
                new_key = key.replace('transformer.layers', 'self_attn_layers')
            elif key.startswith('norm'):
                new_key = key.replace('norm', 'aggregate.3')

            elif key.startswith('value_token'):
                continue
            if new_key and new_key in current_state_dict:
                if current_state_dict[new_key].shape == value.shape:
                    filtered_dict[new_key] = value
                else:
                    print(f"  Shape Dismatch: {key} ({value.shape}) → {new_key} ({current_state_dict[new_key].shape})")
        
        if filtered_dict:
            missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
            print(f"Loaded {len(filtered_dict)} weights from stage 1")
            if missing_keys:
                print(f"  Missing Keys (first 5): {missing_keys[:5]}")
        else:
            print(f"  No weights matched from stage 1!")
            print(f"  Checkpoint key names (first 10): {list(state_dict.keys())[:10]}")
            print(f"  Current model key names (first 10): {list(current_state_dict.keys())[:10]}")


class Discriminator(nn.Module):
    """Output safety score (scalar) for ranking"""
    
    def __init__(self, value_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(value_dim, value_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_dim // 2, 1),  
        )
    
    def forward(self, value_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            value_states: [batch, value_dim] or [batch, seq, value_dim]
        Returns:
            score: [batch] or [batch, seq] (scalar)
        """
        return self.classifier(value_states).squeeze(-1)


class TokenGenerator(nn.Module):
    """
    Generate learnable intervention tokens (based on value transformation)
    """
    
    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.1,
        use_delta_only: bool = True,  
    ):
        super().__init__()
        
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.use_delta_only = use_delta_only
        
        input_dim = value_dim if use_delta_only else value_dim * 2
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * n_tokens),
        )
        
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        
        self.gating_alpha = nn.Parameter(torch.tensor([-2.0]))
        
        # Learnable position embeddings for value tokens
        self.position_embeds = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
        
        # Safe bias: learnable "safety direction" initialization
        # This provides a stable starting point for value tokens
        # delta (from generator) is zero-init, so initially value_tokens ≈ safe_bias
        self.safe_bias = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
    
    def compute_gradient_delta(
        self,
        current_value: torch.Tensor,  # [batch, value_dim]
        discriminator: 'Discriminator',
        step_size: float = 1.0,
        use_relu: bool = True,
    ) -> torch.Tensor:
        """
        Compute delta_value using discriminator gradient
        
        Steps:
        1. Enable gradient computation for current_value
        2. Compute safety score through discriminator
        3. Apply ReLU to only intervene when unsafe (score > 0)
        4. Compute gradient of clamped score w.r.t. current_value
        5. Normalize and scale gradient to get delta_value
        
        Args:
            current_value: [batch, value_dim] current value state
            discriminator: Discriminator instance (frozen during Stage 3)
            step_size: Control intervention strength (default 1.0)
            use_relu: Whether to only intervene when unsafe (default True)
        
        Returns:
            delta_value: [batch, value_dim] value transformation vector
        """
        # Step 1: Enable gradient
        current_value_grad = current_value.clone().detach().requires_grad_(True)
        
        # Step 2: Forward through discriminator
        # Note: discriminator is frozen, but we need gradients w.r.t. current_value
        score = discriminator(current_value_grad)  # [batch]
        
        # Step 3: Clamp to only intervene when unsafe
        if use_relu:
            score_clamped = F.relu(score)  # [batch], only positive (unsafe) scores
        else:
            score_clamped = score
        
        # Step 4: Compute gradient
        if score_clamped.sum() > 0:
            # Only compute gradient when there are unsafe cases
            grad = torch.autograd.grad(
                outputs=score_clamped.sum(),
                inputs=current_value_grad,
                create_graph=False,  # Don't need second-order gradients
                retain_graph=False,
                allow_unused=True,
            )[0]  # [batch, value_dim]
            
            if grad is None:
                grad = torch.zeros_like(current_value_grad)
        else:
            # All safe, no intervention needed
            grad = torch.zeros_like(current_value_grad)
        
        # Step 5: Normalize and scale
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)  # [batch, 1]
        grad_normalized = grad / (grad_norm + 1e-8)  # [batch, value_dim]
        
        # Adaptive step size based on score magnitude
        step_size_adaptive = score_clamped.unsqueeze(-1) / (grad_norm + 1e-8)
        
        # Final delta: negative gradient (move away from unsafe direction)
        delta_v = -step_size_adaptive * grad_normalized * step_size  # [batch, value_dim]
        
        return delta_v.detach()  # Detach to stop gradient flow
    
    def forward(
        self, 
        delta_value: torch.Tensor,
        current_value: torch.Tensor = None,
        discriminator: 'Discriminator' = None,
        use_gradient_delta: bool = False,
        gradient_step_size: float = 1.0,
        trigger_hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate Value Tokens from value transformation
        
        Args:
            delta_value: [batch, value_dim] value transformation (target - current) or placeholder
            current_value: [batch, value_dim] current value state (required if use_gradient_delta=True)
            discriminator: Discriminator instance (required if use_gradient_delta=True)
            use_gradient_delta: Whether to use gradient method to compute delta_value
            gradient_step_size: Step size for gradient method (default 1.0)
            trigger_hidden: [batch, 1, hidden_dim] or [batch, hidden_dim] 
        
        Returns:
            tokens: [batch, n_tokens, hidden_dim]
        """
        # Compute delta_value using gradient method if requested
        if use_gradient_delta:
            if discriminator is None:
                raise ValueError("discriminator is required when use_gradient_delta=True")
            if current_value is None:
                raise ValueError("current_value is required when use_gradient_delta=True")
            delta_value = self.compute_gradient_delta(
                current_value=current_value,
                discriminator=discriminator,
                step_size=gradient_step_size,
                use_relu=True,
            )
        
        if self.use_delta_only:
            input_value = delta_value
        else:
            if current_value is None:
                raise ValueError("current_value is required when use_delta_only=False")
            input_value = torch.cat([current_value, delta_value], dim=-1)  # [batch, value_dim * 2]
        
        # Generate delta (dynamic adjustment based on input)
        delta = self.generator(input_value)  # [batch, hidden_dim * n_tokens]
        delta = delta.view(-1, self.n_tokens, self.hidden_dim)  # [batch, n_tokens, hidden_dim]
        
        # Add learnable position embeddings to delta
        delta = delta + self.position_embeds.unsqueeze(0)  # [batch, n_tokens, hidden_dim]
        
        # Apply gating to delta (controls dynamic adjustment strength)
        gating_factor = F.softplus(self.gating_alpha)
        delta = gating_factor * delta
        
        # Final tokens = safe_bias (fixed direction) + delta (dynamic adjustment)
        # Initially delta ≈ 0 (zero-init), so tokens ≈ safe_bias
        tokens = self.safe_bias.unsqueeze(0) + delta  # [batch, n_tokens, hidden_dim]
        
        if trigger_hidden is not None:
            # trigger_hidden: [batch, 1, hidden_dim] or [batch, hidden_dim]
            if trigger_hidden.dim() == 2:
                # [batch, hidden_dim] -> [batch, 1, hidden_dim]
                trigger_hidden = trigger_hidden.unsqueeze(1)
            # Expand to [batch, n_tokens, hidden_dim]
            trigger_hidden_expanded = trigger_hidden.expand(-1, self.n_tokens, -1)
            tokens = tokens + trigger_hidden_expanded
        
        return tokens


class ValueBridgeGenerator(nn.Module):
    """
    Value-to-Latent Projector (VLP)
    """
    
    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        n_tokens: int = 1,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.value_dim = value_dim
        
        # 1. Query Seeds
        self.query_seeds = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
        
        # 2. Value Projector
        self.value_proj = nn.Sequential(
            nn.Linear(value_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # 4. Gating
        self.gate_alpha = nn.Parameter(torch.tensor([-2.0]))
        
        # 5. LayerNorm 
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def compute_gradient_delta(
        self,
        current_value: torch.Tensor,  # [batch, value_dim]
        discriminator: 'Discriminator',
        step_size: float = 1.0,
        use_relu: bool = True,
    ) -> torch.Tensor:
        """
        Compute delta_value using discriminator gradient
        """
        # Step 1: Enable gradient
        current_value_grad = current_value.clone().detach().requires_grad_(True)
        
        # Step 2: Forward through discriminator
        score = discriminator(current_value_grad)  # [batch]
        
        # Step 3: Clamp to only intervene when unsafe
        if use_relu:
            score_clamped = F.relu(score)  # [batch], only positive (unsafe) scores
        else:
            score_clamped = score
        
        # Step 4: Compute gradient
        if score_clamped.sum() > 0:
            grad = torch.autograd.grad(
                outputs=score_clamped.sum(),
                inputs=current_value_grad,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]  # [batch, value_dim]
            
            if grad is None:
                grad = torch.zeros_like(current_value_grad)
        else:
            # All safe, no intervention needed
            grad = torch.zeros_like(current_value_grad)
        
        # Step 5: Normalize and scale
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)  # [batch, 1]
        grad_normalized = grad / (grad_norm + 1e-8)  # [batch, value_dim]
        
        # Adaptive step size based on score magnitude
        step_size_adaptive = score_clamped.unsqueeze(-1) / (grad_norm + 1e-8)
        
        # Final delta: negative gradient (move away from unsafe direction)
        delta_v = -step_size_adaptive * grad_normalized * step_size  # [batch, value_dim]
        
        return delta_v.detach()  # Detach to stop gradient flow
    
    def forward(
        self,
        h_trigger: torch.Tensor,  # [batch, 1, hidden_dim] 
        delta_z: torch.Tensor,    # [batch, value_dim] 
    ) -> torch.Tensor:
        """
        Generate Value Bridge through Cross-Attention
        
        Args:
            h_trigger: [batch, 1, hidden_dim] 
            delta_z: [batch, value_dim] 
        
        Returns:
            bridge: [batch, n_tokens, hidden_dim] Value Bridge
        """
        B = h_trigger.size(0)
        K = self.n_tokens
        
        v_signal = self.value_proj(delta_z).unsqueeze(1)  # [B, 1, hidden_dim]
        
        kv_context = torch.cat([h_trigger, v_signal], dim=1)  # [B, 2, hidden_dim]
        
        queries = self.query_seeds.unsqueeze(0).expand(B, -1, -1)  # [B, K, hidden_dim]
        attn_out, _ = self.cross_attn(queries, kv_context, kv_context)  # [B, K, hidden_dim]
        
        alpha = F.softplus(self.gate_alpha)  
        trigger_expanded = h_trigger.expand(-1, K, -1)  # [B, K, hidden_dim]
        bridge = self.layer_norm(trigger_expanded + alpha * attn_out)  # [B, K, hidden_dim]
        
        return bridge


class TransformerValueProjector(nn.Module):
    """
    Lightweight Transformer-based Value Projector
    
    Architecture:
    - Value embedding: project value_dim to hidden_dim
    - Learnable query tokens: n_tokens learnable vectors
    - Cross-attention: query tokens attend to value embedding
    - Self-attention: query tokens interact with each other
    - Output projection: zero-initialized output layer
    - Zero-init gating: gating_alpha parameter
    """
    
    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        n_tokens: int = 1,
        n_layers: int = 2,  # Lightweight: 2-3 layers
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        
        # Step 1: Value embedding
        # Project delta_value (or current_value) to hidden_dim
        self.value_embed = nn.Linear(value_dim, hidden_dim)
        nn.init.xavier_uniform_(self.value_embed.weight, gain=0.1)
        nn.init.zeros_(self.value_embed.bias)
        
        # Step 2: Learnable query tokens 
        # These will be the output tokens after processing
        self.query_tokens = nn.Parameter(
            torch.randn(n_tokens, hidden_dim) * 0.02  # Small initialization
        )
        
        # Step 3: Transformer layers
        # Use TransformerDecoderLayer for cross-attention capability
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        
        # Step 4: Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Step 5: Output projection (zero-initialized)
        # Maps from hidden_dim to hidden_dim (for each token)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Step 6: Gating initial
        self.gating_alpha = nn.Parameter(torch.tensor([-2.0]))
        
        # Learnable position embeddings for value tokens
        self.position_embeds = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
    
    def compute_gradient_delta(
        self,
        current_value: torch.Tensor,  # [batch, value_dim]
        discriminator: 'Discriminator',
        step_size: float = 1.0,
        use_relu: bool = True,
    ) -> torch.Tensor:
        """
        Compute delta_value using discriminator gradient
        
        Same implementation as TokenGenerator.compute_gradient_delta
        """
        # Step 1: Enable gradient
        current_value_grad = current_value.clone().detach().requires_grad_(True)
        
        # Step 2: Forward through discriminator
        score = discriminator(current_value_grad)  # [batch]
        
        # Step 3: Clamp to only intervene when unsafe
        if use_relu:
            score_clamped = F.relu(score)  # [batch], only positive (unsafe) scores
        else:
            score_clamped = score
        
        # Step 4: Compute gradient
        if score_clamped.sum() > 0:
            grad = torch.autograd.grad(
                outputs=score_clamped.sum(),
                inputs=current_value_grad,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]  # [batch, value_dim]
            
            if grad is None:
                grad = torch.zeros_like(current_value_grad)
        else:
            grad = torch.zeros_like(current_value_grad)
        
        # Step 5: Normalize and scale
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)  # [batch, 1]
        grad_normalized = grad / (grad_norm + 1e-8)  # [batch, value_dim]
        
        # Adaptive step size based on score magnitude
        step_size_adaptive = score_clamped.unsqueeze(-1) / (grad_norm + 1e-8)
        
        # Final delta: negative gradient (move away from unsafe direction)
        delta_v = -step_size_adaptive * grad_normalized * step_size  # [batch, value_dim]
        
        return delta_v.detach()  # Detach to stop gradient flow
    
    def forward(
        self,
        delta_value: torch.Tensor,  # [batch, value_dim]
        current_value: torch.Tensor = None,  # Optional, for future use
        discriminator: 'Discriminator' = None,
        use_gradient_delta: bool = False,
        gradient_step_size: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate value tokens from delta_value
        
        Process:
        1. Embed delta_value to hidden_dim
        2. Expand query tokens to batch size
        3. Cross-attention: queries attend to value embedding
        4. Self-attention: queries interact
        5. Layer norm
        6. Output projection (zero-init)
        7. Apply gating
        """
        # Compute delta_value using gradient method if requested
        if use_gradient_delta:
            if discriminator is None:
                raise ValueError("discriminator is required when use_gradient_delta=True")
            if current_value is None:
                raise ValueError("current_value is required when use_gradient_delta=True")
            delta_value = self.compute_gradient_delta(
                current_value=current_value,
                discriminator=discriminator,
                step_size=gradient_step_size,
                use_relu=True,
            )
        
        batch_size = delta_value.size(0)
        
        # Step 1: Embed value
        value_emb = self.value_embed(delta_value)  # [batch, hidden_dim]
        value_emb = value_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Step 2: Expand query tokens
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_tokens, hidden_dim]
        
        # Step 3-4: Transformer layers
        # In TransformerDecoderLayer:
        # - tgt (query) attends to memory (value_emb) via cross-attention
        # - tgt also has self-attention
        x = query
        for layer in self.layers:
            x = layer(
                tgt=x,  # Query tokens
                memory=value_emb,  # Value embedding
            )  # [batch, n_tokens, hidden_dim]
        
        # Step 5: Layer norm
        x = self.layer_norm(x)  # [batch, n_tokens, hidden_dim]
        
        # Step 6: Output projection (zero-init)
        tokens = self.output_proj(x)  # [batch, n_tokens, hidden_dim]
        
        # Add learnable position embeddings
        tokens = tokens + self.position_embeds.unsqueeze(0)  # [batch, n_tokens, hidden_dim]
        
        # Step 7: Apply gating
        gating_factor = F.softplus(self.gating_alpha)
        tokens = gating_factor * tokens  # [batch, n_tokens, hidden_dim]
        
        return tokens

