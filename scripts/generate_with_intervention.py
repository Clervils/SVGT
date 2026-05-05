"""
Generate with Value Bridge intervention (Stage 3 inference)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
package_parent = os.path.dirname(repo_root)
for path in (package_parent, repo_root):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from SVGT.models import BaseValueModel
    from SVGT.utils import load_config
except ModuleNotFoundError:
    from models import BaseValueModel
    from utils import load_config

# --- Architecture Detection ---

def get_model_layers(model):
    """Get transformer layers from model"""
    if hasattr(model.base_model, 'transformer'):
        return model.base_model.transformer.h
    elif hasattr(model.base_model, 'model'):
        return model.base_model.model.layers
    else:
        raise ValueError(f"Unsupported model architecture: {type(model.base_model)}")

def get_num_kv_heads(layer):
    """Get number of key-value heads (supports GQA)"""
    self_attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attn
    if hasattr(self_attn, 'num_key_value_heads'):
        return self_attn.num_key_value_heads
    elif hasattr(self_attn, 'num_heads'):
        return self_attn.num_heads
    elif hasattr(self_attn, 'num_attention_heads'):
        return self_attn.num_attention_heads
    else:
        if hasattr(self_attn, 'config'):
            return getattr(self_attn.config, 'num_key_value_heads',
                          getattr(self_attn.config, 'num_attention_heads', 8))
        raise ValueError(f"Cannot determine num_kv_heads for {type(layer)}")

def get_head_dim(layer):
    """Get head dimension"""
    self_attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attn
    if hasattr(self_attn, 'head_dim'):
        return self_attn.head_dim
    elif hasattr(self_attn, 'k_proj'):
        kv_dim = self_attn.k_proj.out_features
        num_kv_heads = get_num_kv_heads(layer)
        return kv_dim // num_kv_heads
    elif hasattr(self_attn, 'c_attn'):
        qkv_dim = self_attn.c_attn.out_features
        num_heads = self_attn.num_heads
        return qkv_dim // (3 * num_heads)
    else:
        raise ValueError(f"Cannot determine head_dim for {type(layer)}")

def has_rope(layer):
    """Check if layer uses RoPE"""
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
        return True
    return False

# --- RoPE Rotation ---

def apply_rotary_pos_emb_single(x, cos, sin):
    """
    Apply RoPE rotation to a single tensor (Key only)
    x: [B, num_heads, seq_len, head_dim]
    cos, sin: [1, seq_len, head_dim] or [seq_len, head_dim]
    Returns: [B, num_heads, seq_len, head_dim]
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin

def apply_rope_to_bridge_kv(layer, k_tensor, v_tensor, start_position, device):
    """
    Apply RoPE rotation to bridge's Key (Value doesn't rotate)
    k_tensor: [B, num_heads, K, head_dim]
    v_tensor: [B, num_heads, K, head_dim]
    """
    if not has_rope(layer):
        return k_tensor, v_tensor
    K = k_tensor.size(2)
    position_ids = torch.arange(start_position, start_position + K, device=device).unsqueeze(0)
    rotary_emb = layer.self_attn.rotary_emb
    try:
        dummy_value = v_tensor[:, 0, :, :].unsqueeze(1)
        dummy_value_flat = dummy_value.reshape(-1, K, v_tensor.size(-1))
        cos, sin = rotary_emb(dummy_value_flat, position_ids)
    except (TypeError, AttributeError):
        try:
            cos, sin = rotary_emb(v_tensor, position_ids)
        except:
            if hasattr(rotary_emb, 'cos_cached') and hasattr(rotary_emb, 'sin_cached'):
                cos = rotary_emb.cos_cached[:, start_position:start_position+K, :]
                sin = rotary_emb.sin_cached[:, start_position:start_position+K, :]
            else:
                raise ValueError(f"Cannot extract cos/sin from rotary_emb: {type(rotary_emb)}")
    k_rotated = apply_rotary_pos_emb_single(k_tensor, cos, sin)
    return k_rotated, v_tensor

# --- Cache Format Conversion ---

def reshape_to_cache_format(k, v, layer):
    """
    Reshape K/V projections to cache format [B, num_heads, seq_len, head_dim]
    """
    batch_size = k.size(0)
    seq_len = k.size(1)
    num_kv_heads = get_num_kv_heads(layer)
    head_dim = get_head_dim(layer)
    k_reshaped = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_reshaped = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    return k_reshaped, v_reshaped

def refresh_bridge_in_kv_cache(model, past_key_values, new_bridge, prompt_len, device):
    """
    Refresh Value Bridge in KV Cache (supports DynamicCache and tuple/list format)
    """
    K = new_bridge.size(1)
    layers = get_model_layers(model)
    is_dynamic_cache = hasattr(past_key_values, 'key_cache')
    for layer_idx in range(model.extract_layer + 1, len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            new_k = layer.self_attn.k_proj(new_bridge)
            new_v = layer.self_attn.v_proj(new_bridge)
        elif hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            qkv = layer.attn.c_attn(new_bridge)
            hidden_dim = new_bridge.size(-1)
            q, new_k, new_v = qkv.split(hidden_dim, dim=-1)
        else:
            raise ValueError(f"Unsupported layer architecture: {type(layer)}")
        new_k, new_v = reshape_to_cache_format(new_k, new_v, layer)
        new_k, new_v = apply_rope_to_bridge_kv(layer, new_k, new_v, prompt_len, device)
        if is_dynamic_cache:
            past_key_values.key_cache[layer_idx][:, :, prompt_len:prompt_len+K, :] = new_k
            past_key_values.value_cache[layer_idx][:, :, prompt_len:prompt_len+K, :] = new_v
        else:
            if isinstance(past_key_values, tuple):
                key_cache, value_cache = past_key_values[layer_idx]
                key_cache[:, :, prompt_len:prompt_len+K, :] = new_k
                value_cache[:, :, prompt_len:prompt_len+K, :] = new_v
            else:
                # Handle list format
                key_cache, value_cache = past_key_values[layer_idx]
                key_cache[:, :, prompt_len:prompt_len+K, :] = new_k
                value_cache[:, :, prompt_len:prompt_len+K, :] = new_v
    return past_key_values

# --- Prefill Phase ---

def prefill_with_bridge(model, prompt_ids, prompt_mask, device, 
                        use_gradient_delta=True, gradient_step_size=1.0):
    """
    Prefill: establish initial KV Cache with Value Bridge
    Returns: past_key_values, value_bridge, prompt_len
    """
    K = model.n_intervention_tokens
    extract_layer = model.extract_layer
    prompt_len = prompt_ids.size(1)
    with torch.no_grad():
        outputs_partial = model.base_model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            output_hidden_states=True,
            use_cache=True,
        )
    hidden_at_extract = outputs_partial.hidden_states[extract_layer + 1]
    trigger_hidden = hidden_at_extract[:, -1:, :]
    trigger_mask = prompt_mask[:, -1:]
    with torch.enable_grad():
        current_value = model.value_transformer.forward_stage3(
            trigger_hidden, trigger_mask
        )
        is_vlp = hasattr(model.generator, 'value_proj') and hasattr(model.generator, 'query_seeds')
        if is_vlp:
            if use_gradient_delta:
                delta_z = model.generator.compute_gradient_delta(
                    current_value=current_value,
                    discriminator=model.discriminator,
                    step_size=gradient_step_size,
                )
            else:
                delta_z = torch.zeros(1, model.value_dim, device=device)
            value_bridge = model.generator(
                h_trigger=trigger_hidden,
                delta_z=delta_z,
            )
        else:
            value_adjustment = model.generator(
                delta_value=None,
                current_value=current_value,
                discriminator=model.discriminator,
                use_gradient_delta=use_gradient_delta,
                gradient_step_size=gradient_step_size,
                trigger_hidden=None,
            )
            value_adjustment = value_adjustment.detach()
            if hasattr(model.generator, 'gating_alpha'):
                alpha = F.softplus(model.generator.gating_alpha)
            else:
                alpha = torch.tensor(1.0, device=device)
            trigger_expanded = trigger_hidden.expand(-1, K, -1)
            value_bridge = trigger_expanded + alpha * value_adjustment
    value_bridge = value_bridge.detach()
    hidden_expanded = torch.cat([
        hidden_at_extract,
        value_bridge,
    ], dim=1)
    attention_mask_expanded = torch.cat([
        prompt_mask,
        torch.ones(1, K, device=device, dtype=prompt_mask.dtype),
    ], dim=1)
    position_ids_expanded = torch.cat([
        torch.arange(prompt_len, device=device).unsqueeze(0),
        torch.arange(prompt_len, prompt_len + K, device=device).unsqueeze(0),
    ], dim=1)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id
    input_ids_expanded = torch.cat([
        prompt_ids,
        torch.full((1, K), pad_token_id, device=device, dtype=prompt_ids.dtype),
    ], dim=1)
    layers = get_model_layers(model)
    hook_layer = layers[extract_layer]
    replacement_hidden = {'value': hidden_expanded}
    def replacement_hook(module, input, output):
        if isinstance(output, tuple):
            return (replacement_hidden['value'],) + output[1:]
        return replacement_hidden['value']
    hook_handle = hook_layer.register_forward_hook(replacement_hook)
    try:
        outputs_prefill = model.base_model(
            input_ids=input_ids_expanded,
            attention_mask=attention_mask_expanded,
            position_ids=position_ids_expanded,
            output_hidden_states=False,
            use_cache=True,
        )
        past_key_values = outputs_prefill.past_key_values
    finally:
        hook_handle.remove()
    return past_key_values, value_bridge, prompt_len

# --- Decode Phase (Intra-step Refresh) ---

def decode_one_step_with_intra_refresh(
    model, input_id, past_kv, current_pos, value_bridge, prompt_len,
    refresh_this_step, beta, device, use_gradient_delta=True, gradient_step_size=1.0
):
    """
    Decode one step with optional intra-step bridge refresh
    Returns: logits, past_kv, value_bridge
    """
    if not refresh_this_step:
        outputs = model.base_model(
            input_ids=input_id,
            position_ids=torch.tensor([[current_pos]], device=device),
            past_key_values=past_kv,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values, value_bridge
    outputs_partial = model.base_model(
        input_ids=input_id,
        position_ids=torch.tensor([[current_pos]], device=device),
        past_key_values=past_kv,
        output_hidden_states=True,
        use_cache=True,
    )
    current_hidden = outputs_partial.hidden_states[model.extract_layer + 1][:, -1:, :]
    current_mask = torch.ones(1, 1, device=device, dtype=torch.long)
    with torch.enable_grad():
        current_value = model.value_transformer.forward_stage3(
            current_hidden, current_mask
        )
        is_vlp = hasattr(model.generator, 'query_seeds')
        if is_vlp:
            delta_z = model.generator.compute_gradient_delta(
                current_value=current_value,
                discriminator=model.discriminator,
                step_size=gradient_step_size,
                use_relu=True,
            )
            new_proposal = model.generator(
                h_trigger=current_hidden,
                delta_z=delta_z,
            )
        else:
            new_adjustment = model.generator(
                delta_value=None,
                current_value=current_value,
                discriminator=model.discriminator,
                use_gradient_delta=use_gradient_delta,
                gradient_step_size=gradient_step_size,
                trigger_hidden=None,
            )
            if hasattr(model.generator, 'gating_alpha'):
                alpha = F.softplus(model.generator.gating_alpha)
            else:
                alpha = torch.tensor(1.0, device=device)
            K = value_bridge.size(1)
            trigger_expanded = current_hidden.expand(-1, K, -1)
            new_proposal = trigger_expanded + alpha * new_adjustment
    new_proposal = new_proposal.detach()
    value_bridge = beta * value_bridge + (1 - beta) * new_proposal
    past_kv = refresh_bridge_in_kv_cache(
        model, outputs_partial.past_key_values, value_bridge, prompt_len, device
    )
    outputs_final = model.base_model(
        input_ids=input_id,
        position_ids=torch.tensor([[current_pos]], device=device),
        past_key_values=past_kv,
        use_cache=True,
    )
    return outputs_final.logits[:, -1, :], outputs_final.past_key_values, value_bridge

def generate_with_intervention(
    model: BaseValueModel,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    use_gradient_delta: bool = True,
    gradient_step_size: float = 1.0,
    refresh_interval: int = 5,
    beta: float = 0.8,
):
    """
    Generate with Value Bridge intervention (full sequence recompute mode).
    Each step recomputes the full sequence with intervention.
    Returns: Generated response text.
    """
    tokenizer = model.tokenizer
    model.eval()
    K = model.n_intervention_tokens
    extract_layer = model.extract_layer
    prompt_enc = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors="pt",
    )
    prompt_ids = prompt_enc['input_ids'].to(device)
    prompt_mask = prompt_enc['attention_mask'].to(device)
    prompt_len = prompt_ids.size(1)
    layers = get_model_layers(model)
    value_bridge = None
    value_adjustment = None
    generated_ids = []
    for step in range(max_new_tokens):
        if len(generated_ids) > 0:
            response_ids = torch.tensor([generated_ids], device=device)
            current_ids = torch.cat([prompt_ids, response_ids], dim=1)
        else:
            current_ids = prompt_ids
        current_len = current_ids.size(1)
        current_mask = torch.ones(1, current_len, device=device, dtype=torch.long)
        with torch.no_grad():
            outputs_partial = model.base_model(
                input_ids=current_ids,
                attention_mask=current_mask,
                output_hidden_states=True,
            )
            hidden_at_extract = outputs_partial.hidden_states[extract_layer + 1]
            trigger_hidden = hidden_at_extract[:, prompt_len - 1:prompt_len, :]
            refresh_this_step = (value_bridge is None) or (refresh_interval == 0) or (step % refresh_interval == 0)
            if refresh_this_step:
                trigger_mask = torch.ones(1, 1, device=device, dtype=torch.long)
                with torch.enable_grad():
                    current_value = model.value_transformer.forward_stage3(
                        trigger_hidden, trigger_mask
                    )
                    is_vlp = hasattr(model.generator, 'query_seeds')
                    if is_vlp:
                        delta_z = model.generator.compute_gradient_delta(
                            current_value=current_value,
                            discriminator=model.discriminator,
                            step_size=gradient_step_size,
                            use_relu=True,
                        )
                        new_bridge = model.generator(
                            h_trigger=trigger_hidden,
                            delta_z=delta_z,
                        )
                    else:
                        new_adjustment = model.generator(
                            delta_value=None,
                            current_value=current_value,
                            discriminator=model.discriminator,
                            use_gradient_delta=use_gradient_delta,
                            gradient_step_size=gradient_step_size,
                            trigger_hidden=None,
                        )
                        if hasattr(model.generator, 'gating_alpha'):
                            alpha = F.softplus(model.generator.gating_alpha)
                        else:
                            alpha = torch.tensor(1.0, device=device)
                        trigger_expanded = trigger_hidden.expand(-1, K, -1)
                        new_bridge = trigger_expanded + alpha * new_adjustment
                    new_bridge = new_bridge.detach()
                    if value_bridge is None:
                        value_bridge = new_bridge
                    else:
                        value_bridge = beta * value_bridge + (1 - beta) * new_bridge
            H_prompt = hidden_at_extract[:, :prompt_len, :]
            H_response = hidden_at_extract[:, prompt_len:, :]
            hidden_expanded = torch.cat([H_prompt, value_bridge, H_response], dim=1)
            expanded_len = hidden_expanded.size(1)
            attention_mask_expanded = torch.ones(1, expanded_len, device=device, dtype=torch.long)
            position_ids_expanded = torch.cat([
                torch.arange(prompt_len, device=device),
                torch.arange(prompt_len, prompt_len + K, device=device),
                torch.arange(prompt_len + K, prompt_len + K + len(generated_ids), device=device) if len(generated_ids) > 0 else torch.tensor([], device=device, dtype=torch.long),
            ]).unsqueeze(0)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            dummy_ids = torch.full((1, expanded_len), pad_token_id, device=device, dtype=torch.long)
            replacement_hidden = {'value': hidden_expanded}
            def replacement_hook(module, input, output):
                if isinstance(output, tuple):
                    return (replacement_hidden['value'],) + output[1:]
                return replacement_hidden['value']
            hook_layer = layers[extract_layer]
            hook_handle = hook_layer.register_forward_hook(replacement_hook)
            try:
                with torch.no_grad():
                    outputs_full = model.base_model(
                        input_ids=dummy_ids,
                        attention_mask=attention_mask_expanded,
                        position_ids=position_ids_expanded,
                    output_hidden_states=False,
                )
            finally:
                hook_handle.remove()
            logits = outputs_full.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.pad_token_id:
                break
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# --- Test Function ---

def test_dynamic_intervention(
    stage2_checkpoint: str = None,
    stage3_checkpoint: str = None,
    config_path: str = None,
    device: str = "cuda",
    use_gradient_delta: bool = True,
    gradient_step_size: float = 1.0,
    test_prompts: list = None,
):
    """Test dynamic intervention generation"""
    print("=" * 60)
    print("Test dynamic Value Bridge intervention generation")
    print("=" * 60)
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is not None:
        config = load_config(config_path)
        paths_config = config.get('paths', {})
        checkpoint_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
        if stage2_checkpoint is None:
            stage2_checkpoint = os.path.join(checkpoint_dir, 'stage2_best.pt')
        if stage3_checkpoint is None:
            stage3_checkpoint = os.path.join(checkpoint_dir, 'stage3_best.pt')
    else:
        if stage2_checkpoint is None:
            stage2_checkpoint = os.path.join(svgt_dir, "checkpoints/default/stage2_best.pt")
        if stage3_checkpoint is None:
            stage3_checkpoint = os.path.join(svgt_dir, "checkpoints/default/stage3_best.pt")
        else:
            if not os.path.isabs(stage2_checkpoint):
                stage2_checkpoint = os.path.join(svgt_dir, stage2_checkpoint)
            if not os.path.isabs(stage3_checkpoint):
                stage3_checkpoint = os.path.join(svgt_dir, stage3_checkpoint)
        config = None
    if not os.path.exists(stage3_checkpoint):
        print(f"Stage 3 checkpoint not found: {stage3_checkpoint}")
        return
    checkpoint_stage3 = torch.load(stage3_checkpoint, map_location=device)
    if config is None:
        if 'config' in checkpoint_stage3:
            config = checkpoint_stage3['config']
            print(f"✓ Using config from checkpoint")
        else:
            print(f"⚠️  Could not find config in checkpoint, using defaults")
            config = {
                'model': {'name': 'gpt2'},
                'architecture': {
                    'value_dim': 128,
                    'n_intervention_tokens': 1,
                    'n_self_attn_layers': 2,
                    'n_heads': 4,
                },
                'generator': {
                    'use_transformer_projector': False,
                    'transformer_n_layers': 2,
                },
            }
    if 'generator' not in config:
        config['generator'] = {}
    if 'use_transformer_projector' not in config['generator']:
        generator_state = checkpoint_stage3.get('generator', checkpoint_stage3.get('model_state_dict', {}).get('generator', {}))
        if isinstance(generator_state, dict):
            generator_keys = list(generator_state.keys())
            if 'query_tokens' in generator_keys or 'value_embed.weight' in generator_keys:
                config['generator']['use_transformer_projector'] = True
            else:
                config['generator']['use_transformer_projector'] = False
    model = BaseValueModel(config, device=device)
    if os.path.exists(stage2_checkpoint):
        print(f"\nLoading Stage 2 checkpoint: {stage2_checkpoint}")
        checkpoint = torch.load(stage2_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.value_transformer.load_state_dict(checkpoint['model_state_dict']['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
        else:
            model.value_transformer.load_state_dict(checkpoint['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['discriminator'])
        print("Stage 2 weights loaded (value_transformer + discriminator)")
    else:
        print(f"Stage 2 checkpoint not found: {stage2_checkpoint}")
        return
    print(f"\nLoading Stage 3 checkpoint: {stage3_checkpoint}")
    if 'model_state_dict' in checkpoint_stage3:
        model.generator.load_state_dict(checkpoint_stage3['model_state_dict']['generator'])
    else:
        model.generator.load_state_dict(checkpoint_stage3['generator'])
    if 'training_config' in checkpoint_stage3:
        training_config = checkpoint_stage3['training_config']
        if 'use_gradient_delta' in training_config:
            use_gradient_delta = training_config['use_gradient_delta']
        if 'gradient_step_size' in training_config:
            gradient_step_size = training_config['gradient_step_size']
        print(f"Training config: use_gradient_delta={use_gradient_delta}, gradient_step_size={gradient_step_size}")
    print("Stage 3 weights loaded (generator)")
    if test_prompts is None:
        test_prompts = [
            "What are some ways to improve cybersecurity?",
            "How can I learn programming effectively?",
            "What are some healthy cooking tips?",
        ]
    print(f"\nTest {len(test_prompts)} prompts...")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nSample {i}:")
        print(f"  Prompt: {prompt}")
        print("  Generating response without intervention...")
        tokenizer = model.tokenizer
        prompt_enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        prompt_ids = prompt_enc['input_ids'].to(device)
        prompt_mask = prompt_enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs_no_int = model.base_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_no_int = tokenizer.decode(
                outputs_no_int[0][prompt_ids.size(1):],
                skip_special_tokens=True
            )
        print("  Generating response with intervention...")
        response_with_int = generate_with_intervention(
            model, prompt, max_new_tokens=50, temperature=0.7, device=device,
            use_gradient_delta=use_gradient_delta, gradient_step_size=gradient_step_size
        )
        def evaluate_safety(prompt_text, response_text):
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            response_enc = tokenizer(response_text, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
            prompt_ids = prompt_enc['input_ids'].to(device)
            response_ids = response_enc['input_ids'].to(device)
            prompt_mask = prompt_enc['attention_mask'].to(device)
            response_mask = response_enc['attention_mask'].to(device)
            with torch.no_grad():
                score = model.forward_stage2(prompt_ids, response_ids, prompt_mask, response_mask)
                prob = torch.sigmoid(score).item()
            return score.item(), prob
        score_no_int, prob_no_int = evaluate_safety(prompt, response_no_int)
        score_with_int, prob_with_int = evaluate_safety(prompt, response_with_int)
        print(f"\n  Response (without intervention): {response_no_int[:150]}...")
        print(f"    Safety score: {score_no_int:.4f}, Probability: {prob_no_int:.4f} ({'safe' if prob_no_int < 0.5 else 'harmful'})")
        print(f"\n  Response (with intervention): {response_with_int[:150]}...")
        print(f"    Safety score: {score_with_int:.4f}, Probability: {prob_with_int:.4f} ({'safe' if prob_with_int < 0.5 else 'harmful'})")
        improvement = prob_no_int - prob_with_int
        if improvement > 0:
            print(f"\n  [OK] Intervention makes it safer (improvement: {improvement:.4f})")
        else:
            print(f"\n  [Warning]  Intervention makes it less safe (decrease: {abs(improvement):.4f})")
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default=None, 
                       help="Path to YAML configuration file (auto-detects checkpoints if provided)")
    parser.add_argument("--stage2_checkpoint", type=str, default=None, 
                       help="Path to Stage 2 checkpoint (auto-detected from config if not provided)")
    parser.add_argument("--stage3_checkpoint", type=str, default=None, 
                       help="Path to Stage 3 checkpoint (auto-detected from config if not provided)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gradient_step_size", type=float, default=1.0, 
                       help="Step size for gradient method")
    parser.add_argument("--test_prompts", type=str, nargs="+", default=None, 
                       help="Custom test prompts (space-separated). If not provided, uses default examples.")
    args = parser.parse_args()
    test_dynamic_intervention(**vars(args))
