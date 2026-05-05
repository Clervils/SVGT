"""
BaseValueModel: Unified value alignment intervention system
Supports different backbone models through configuration
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict

from .value_stream import (
    ValueTransformer, 
    Discriminator, 
    TokenGenerator,
    TransformerValueProjector,
    ValueBridgeGenerator,
)


class BaseValueModel(nn.Module):
    """Unified value alignment intervention system base class"""
    
    def __init__(self, config: Dict, device: str = "cuda"):
        """
        Initialize model from configuration dictionary
        
        Args:
            config: Configuration dictionary with keys:
                - model: {'name': str}
                - architecture: {value_dim, n_intervention_tokens, extract_layer, ...}
                - generator: {use_transformer_projector, transformer_n_layers}
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.config = config
        
        # Extract configuration
        model_config = config.get('model', {})
        arch_config = config.get('architecture', {})
        gen_config = config.get('generator', {})
        
        self.model_name = model_config.get('name', 'gpt2')
        self.value_dim = arch_config.get('value_dim', 128)
        
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            device_id = int(device.split(':')[1]) if ':' in device else 0
            torch.cuda.set_device(device_id)

        load_kwargs = {
            'low_cpu_mem_usage': True,
        }
        is_large_model = any(size in self.model_name.lower() for size in ['3b', '7b', '8b', '13b', '70b'])
        
        if is_large_model:
            target_device = device
            if device == "cuda":
                target_device = "cuda:0"
            load_kwargs.update({
                'torch_dtype': torch.bfloat16,
                'device_map': {'': target_device},
            })
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        if 'device_map' not in load_kwargs:
            self.base_model = self.base_model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.base_model.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False

        config_obj = self.base_model.config
        if hasattr(config_obj, 'n_embd'):  # GPT-2 style
            self.hidden_dim = config_obj.n_embd
        elif hasattr(config_obj, 'hidden_size'):  # Qwen/Llama style
            self.hidden_dim = config_obj.hidden_size
        else:
            raise ValueError(f"Unknown model config: {config_obj}")
        
        if hasattr(config_obj, 'n_head'):
            self.n_head = config_obj.n_head
        elif hasattr(config_obj, 'num_attention_heads'):
            self.n_head = config_obj.num_attention_heads
        else:
            self.n_head = arch_config.get('n_heads', 4)
        
        if hasattr(config_obj, 'n_layer'):
            self.n_layer = config_obj.n_layer
        elif hasattr(config_obj, 'num_hidden_layers'):
            self.n_layer = config_obj.num_hidden_layers
        else:
            raise ValueError(f"Unknown model config: {config_obj}")

        extract_layer = arch_config.get('extract_layer')
        if extract_layer is None:
            if self.n_layer > 20:
                extract_layer = self.n_layer // 2
            else:
                extract_layer = self.n_layer - 3
        
        self.extract_layer = extract_layer
        self.n_intervention_tokens = arch_config.get('n_intervention_tokens', 1)
        
        # Value transformer
        self.value_transformer = ValueTransformer(
            hidden_dim=self.hidden_dim,
            value_dim=self.value_dim,
            n_self_attn_layers=arch_config.get('n_self_attn_layers', 2),
            n_heads=arch_config.get('n_heads', 4),
            dropout=arch_config.get('dropout', 0.1),
            use_attention_pooling=arch_config.get('use_attention_pooling', False),
            use_transformer_aggregate=arch_config.get('use_transformer_aggregate', False),
        )
        
        # Discriminator
        self.discriminator = Discriminator(
            self.value_dim,
            arch_config.get('dropout', 0.1)
        )
        
        # Generator
        use_vlp = gen_config.get('use_vlp', False)
        if use_vlp:
            self.generator = ValueBridgeGenerator(
                value_dim=self.value_dim,
                hidden_dim=self.hidden_dim,
                n_tokens=self.n_intervention_tokens,
                n_heads=gen_config.get('vlp_n_heads', 8),
                dropout=arch_config.get('dropout', 0.1),
            )
        elif gen_config.get('use_transformer_projector', False):
            self.generator = TransformerValueProjector(
                value_dim=self.value_dim,
                hidden_dim=self.hidden_dim,
                n_tokens=self.n_intervention_tokens,
                n_layers=gen_config.get('transformer_n_layers', 2),
                n_heads=arch_config.get('n_heads', 4),
                dropout=arch_config.get('dropout', 0.1),
            )
        else:
            self.generator = TokenGenerator(
                value_dim=self.value_dim,
                hidden_dim=self.hidden_dim,
                n_tokens=self.n_intervention_tokens,
                dropout=arch_config.get('dropout', 0.1),
                use_delta_only=True,
            )

        if 'device_map' not in load_kwargs:
            self.base_model = self.base_model.to(device)
        
        base_dtype = next(self.base_model.parameters()).dtype
        
        self.value_transformer = self.value_transformer.to(device).to(base_dtype)
        self.discriminator = self.discriminator.to(device).to(base_dtype)
        self.generator = self.generator.to(device).to(base_dtype)
        
        # Print initialization info
        self._print_init_info(arch_config, gen_config)
    
    def _print_init_info(self, arch_config, gen_config):
        """Print model initialization information"""
        print(f"BaseValueModel initialized:")
        print(f"  Base model: {self.model_name}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Value dim: {self.value_dim}")
        print(f"  N intervention tokens: {self.n_intervention_tokens}")
        print(f"  Extract layer: {self.extract_layer}")
        print(f"  Self-Attention layers: {arch_config.get('n_self_attn_layers', 2)}")
        print(f"  Attention heads: {arch_config.get('n_heads', 4)}")
        if arch_config.get('use_attention_pooling', False):
            print(f"  Use attention pooling: True")
        if arch_config.get('use_transformer_aggregate', False):
            print(f"  Use transformer aggregate: True")
        if gen_config.get('use_vlp', False):
            print(f"  Generator: ValueBridgeGenerator (VLP, n_heads={gen_config.get('vlp_n_heads', 8)})")
        elif gen_config.get('use_transformer_projector', False):
            print(f"  Generator: TransformerValueProjector (n_layers={gen_config.get('transformer_n_layers', 2)})")
        else:
            print(f"  Generator: TokenGenerator (MLP-based)")
        
        trainable = sum(p.numel() for p in self.value_transformer.parameters()) + \
                   sum(p.numel() for p in self.discriminator.parameters()) + \
                   sum(p.numel() for p in self.generator.parameters())
        print(f"  Trainable params: {trainable:,}")
    
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Extract hidden states from base model at the specified layer
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            requires_grad: Whether to enable gradients (for training)
        
        Returns:
            Hidden states tensor [batch_size, seq_len, hidden_dim] at extract_layer
        """
        backbone = None
        if hasattr(self.base_model, "model"):
            backbone = self.base_model.model
        elif hasattr(self.base_model, "transformer"):
            backbone = self.base_model.transformer

        def forward_hidden():
            target = backbone if backbone is not None else self.base_model
            return target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        if not requires_grad:
            with torch.no_grad():
                outputs = forward_hidden()
        else:
            outputs = forward_hidden()
        
        hidden_states = outputs.hidden_states[self.extract_layer + 1]
        return hidden_states
    
    def forward_stage1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 1: Unconditional value representation learning
        
        Learns value representation from response text only, without prompt context.
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
        
        Returns:
            Safety score tensor [batch_size, 1]
        """
        hidden_states = self.get_hidden_states(input_ids, attention_mask, requires_grad=True)
        z = self.value_transformer.forward_unconditional(hidden_states, attention_mask)
        score = self.discriminator(z)
        return score
    
    def forward_stage2(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 2: Conditional value representation learning
        
        Learns value representation from prompt-response pairs using cross-attention.
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            response_ids: Response token IDs [batch_size, response_len]
            prompt_mask: Prompt attention mask [batch_size, prompt_len]
            response_mask: Response attention mask [batch_size, response_len]
        
        Returns:
            Safety score tensor [batch_size, 1]
        """
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        
        full_hidden = self.get_hidden_states(full_ids, full_mask, requires_grad=True)
        
        prompt_hidden = full_hidden[:, :prompt_len, :]
        response_hidden = full_hidden[:, prompt_len:, :]
        
        z = self.value_transformer.forward_conditional(
            prompt_hidden, response_hidden,
            prompt_mask, response_mask
        )
        
        score = self.discriminator(z)
        return score
    
    def freeze_for_stage1(self):
        """
        Freeze/unfreeze parameters for Stage 1 training
        """
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.value_transformer.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True
    
    def freeze_for_stage2(self):
        """
        Freeze/unfreeze parameters for Stage 2 training
        """
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.value_transformer.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True
    
    def get_stage2_parameter_groups(self, lr_new: float = 1e-4, lr_finetune: float = 1e-5):
        """Get parameter groups for stage 2 training"""
        new_params = []
        new_param_names = ['prompt_proj', 'cross_attn', 'cross_norm', 'cross_scale']
        
        finetune_params = []
        finetune_param_names = ['response_proj', 'self_attn_layers', 'aggregate']
        
        classifier_params = list(self.discriminator.parameters())
        
        for name, param in self.value_transformer.named_parameters():
            if any(name.startswith(prefix) for prefix in new_param_names):
                new_params.append(param)
            elif any(name.startswith(prefix) for prefix in finetune_param_names):
                finetune_params.append(param)
        
        parameter_groups = [
            {'params': new_params, 'lr': lr_new, 'name': 'new_components'},
            {'params': finetune_params, 'lr': lr_finetune, 'name': 'finetune_components'},
            {'params': classifier_params, 'lr': lr_finetune, 'name': 'classifier'},
        ]
        
        return parameter_groups
    
    def forward_stage3(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract prompt value representation for Stage 3 intervention
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            prompt_mask: Prompt attention mask [batch_size, prompt_len]
        
        Returns:
            Prompt value representation [batch_size, value_dim]
        """
        prompt_hidden = self.get_hidden_states(prompt_ids, prompt_mask, requires_grad=True)
        prompt_value = self.value_transformer.forward_stage3(prompt_hidden, prompt_mask)
        return prompt_value
    
    def generate_value_tokens(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        current_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate intervention value tokens from prompt
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            prompt_mask: Prompt attention mask [batch_size, prompt_len]
            current_hidden_states: Optional pre-computed hidden states [batch_size, seq_len, hidden_dim]
                If provided, uses these instead of recomputing
        
        Returns:
            Value tokens tensor [batch_size, n_tokens, hidden_dim] for intervention
        """
        if current_hidden_states is not None:
            prompt_len = prompt_ids.size(1) if prompt_ids.size(1) <= current_hidden_states.size(1) else current_hidden_states.size(1)
            prompt_hidden = current_hidden_states[:, :prompt_len, :]
            prompt_value = self.value_transformer.forward_stage3(prompt_hidden, prompt_mask)
        else:
            prompt_value = self.forward_stage3(prompt_ids, prompt_mask)
        
        value_tokens = self.generator(prompt_value)
        return value_tokens
    
    def freeze_for_stage3(self, training_phase: str = "unified"):
        """
        Freeze/unfreeze parameters for Stage 3 training
        
        Args:
            training_phase: Training phase mode (kept for backward compatibility, but always uses unified behavior)
                - "unified": Freeze value transformer, train generator only
                - "phase1": Freeze value transformer, train generator only (same as unified)
                - "phase2": Train both value transformer and generator (deprecated, now same as unified)
        """
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Freeze value transformer
        for param in self.value_transformer.parameters():
            param.requires_grad = False
        
        # Freeze discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        # Train generator
        for param in self.generator.parameters():
            param.requires_grad = True
        
        # Train gating alpha (if exists)
        if hasattr(self.generator, 'gating_alpha'):
            self.generator.gating_alpha.requires_grad = True
    
    def get_stage3_parameter_groups(self, lr_new: float = 5e-4, lr_finetune: float = 1e-5):
        """
        Get parameter groups for stage 3 training
        
        Args:
            lr_new: Learning rate for generator parameters
            lr_finetune: Learning rate for gating alpha (if exists, otherwise unused)
        """
        generator_params = []
        gating_params = []
        
        for name, param in self.generator.named_parameters():
            if 'gating_alpha' in name or 'gate_alpha' in name:
                gating_params.append(param)
            else:
                generator_params.append(param)
        
        parameter_groups = [
            {'params': generator_params, 'lr': lr_new, 'name': 'generator'},
        ]
        
        if gating_params:
            parameter_groups.append({
                'params': gating_params, 
                'lr': lr_new,  
                'name': 'gating_alpha'
            })
        
        return parameter_groups
