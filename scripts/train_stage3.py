"""Stage 3: Value token intervention generation training"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
package_parent = os.path.dirname(repo_root)
for path in (package_parent, repo_root):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from SVGT.training.stage3_loader import Stage3Dataset
    from SVGT.models import BaseValueModel
    from SVGT.utils import load_config, parse_cli_overrides
except ModuleNotFoundError:
    from training.stage3_loader import Stage3Dataset
    from models import BaseValueModel
    from utils import load_config, parse_cli_overrides


def filter_safe_only(data_path):
    """Filter safe-only samples (deprecated, kept for compatibility)"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item for item in data if item.get('response_is_harmful', True) == False]


def load_paired_data(data_path):
    """
    Load data with both safe and harmful responses.
    Each sample should have 'prompt' and 'response' fields.
    If response_is_harmful is True, the response is harmful.
    We group by prompt to find matching safe/harmful pairs.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by prompt
    prompt_to_samples = {}
    for item in data:
        prompt = item.get('prompt', '')
        if prompt not in prompt_to_samples:
            prompt_to_samples[prompt] = {'safe': None, 'harmful': None}
        
        is_harmful = item.get('response_is_harmful', False)
        if is_harmful:
            prompt_to_samples[prompt]['harmful'] = item.get('response', '')
        else:
            prompt_to_samples[prompt]['safe'] = item.get('response', '')
    
    # Create paired samples
    paired_data = []
    for prompt, responses in prompt_to_samples.items():
        if responses['safe']:  # Must have safe response
            paired_data.append({
                'prompt': prompt,
                'response': responses['safe'],
                'response_is_harmful': False,
                'harmful_response': responses.get('harmful', '') 
            })
    
    return paired_data


def detect_architecture(model):
    if hasattr(model.base_model, 'transformer'):
        arch_type = "GPT-2"
        transformer = model.base_model.transformer
        layers_attr = 'h'
    elif hasattr(model.base_model, 'model'):
        arch_type = "Llama/Mistral/Qwen"
        transformer = model.base_model.model
        layers_attr = 'layers'
    else:
        raise ValueError(f"Unsupported architecture: {type(model.base_model)}")
    
    layers = getattr(transformer, layers_attr)
    num_layers = len(layers)
    
    return arch_type, num_layers


def compute_losses_detailed_sft(
    model,
    input_ids: torch.Tensor,           # [Prompt, Response, EOS] 
    labels: torch.Tensor,               # [-100(Prompt), Response, EOS]
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,         # [0..M-1, M+K..M+K+L] 
    prompt_len: torch.Tensor,           # [batch] 
    device: str = "cuda",
    use_gradient_delta: bool = True,
    gradient_step_size: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_safe: float = 0.5,
    lambda_reg: float = 0.1,
) -> dict:
    
    K = model.n_intervention_tokens
    extract_layer = model.extract_layer
    batch_size = input_ids.size(0)
    seq_len_original = input_ids.size(1) 
    
    with torch.no_grad():
        hidden_states_at_extract = run_partial_forward(
            model, input_ids, attention_mask, position_ids, 
            stop_at_layer=extract_layer, requires_grad=False
        )
    
    batch_size, seq_len, hidden_dim = hidden_states_at_extract.shape
    expanded_seq_len = seq_len + K
    M_list = prompt_len.tolist()  
    
    trigger_list = []
    for b in range(batch_size):
        M = int(M_list[b])
        if M > 0:
            trigger = hidden_states_at_extract[b, M-1:M, :]  # [1, D]
        else:
            trigger = hidden_states_at_extract[b, 0:1, :]
        trigger_list.append(trigger)
    
    trigger_hidden = torch.stack(trigger_list, dim=0).squeeze(1).unsqueeze(1)  # [B, 1, D]
    trigger_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
    
    current_value = model.value_transformer.forward_stage3(trigger_hidden, trigger_mask)  # [B, value_dim]
    is_vlp = hasattr(model.generator, 'value_proj') and hasattr(model.generator, 'query_seeds')
    
    if is_vlp:
        if use_gradient_delta:
            delta_z = model.generator.compute_gradient_delta(
                current_value=current_value,
                discriminator=model.discriminator,
                step_size=gradient_step_size,
            )  # [B, value_dim]
        else:
            delta_z = torch.zeros(batch_size, model.value_dim, device=device)
    
        value_bridge = model.generator(
            h_trigger=trigger_hidden,  # [B, 1, D]
            delta_z=delta_z,           # [B, value_dim]
        )  # [B, K, D]
    else:
        value_adjustment = model.generator(
            delta_value=None,
            current_value=current_value,
            discriminator=model.discriminator,
            use_gradient_delta=use_gradient_delta,
            gradient_step_size=gradient_step_size,
            trigger_hidden=None,  
        )  # [B, K, D]
        
        if hasattr(model.generator, 'gating_alpha'):
            alpha = F.softplus(model.generator.gating_alpha) 
        else:
            alpha = torch.tensor(1.0, device=device)
        
        trigger_expanded = trigger_hidden.expand(-1, K, -1)  # [B, K, D]
        value_bridge = trigger_expanded + alpha * value_adjustment  # [B, K, D]
    
    H_expanded_list = []
    position_ids_full_list = []
    attention_mask_expanded_list = []
    
    for b in range(batch_size):
        M = int(M_list[b])

        H_prompt_b = hidden_states_at_extract[b, :M, :]  # [M, D]
        H_response_b = hidden_states_at_extract[b, M:, :]  # [seq_len - M, D]
        value_bridge_b = value_bridge[b]  # [K, D]
        
        H_expanded_b = torch.cat([H_prompt_b, value_bridge_b, H_response_b], dim=0)  # [seq_len + K, D]
        H_expanded_list.append(H_expanded_b)

        pos_prompt_b = position_ids[b, :M]  # [M]
        pos_response_b = position_ids[b, M:]  # [seq_len - M]
        pos_bridge_b = torch.arange(M, M + K, dtype=torch.long, device=device)  # [K]
        position_ids_full_b = torch.cat([pos_prompt_b, pos_bridge_b, pos_response_b], dim=0)  # [seq_len + K]
        position_ids_full_list.append(position_ids_full_b)

        mask_prompt_b = attention_mask[b, :M]  # [M]
        mask_response_b = attention_mask[b, M:]  # [seq_len - M]
        mask_bridge_b = torch.ones(K, dtype=torch.long, device=device)  # [K]
        attention_mask_expanded_b = torch.cat([mask_prompt_b, mask_bridge_b, mask_response_b], dim=0)  # [seq_len + K]
        attention_mask_expanded_list.append(attention_mask_expanded_b)

    H_expanded = torch.stack(H_expanded_list, dim=0)  # [B, seq_len + K, D]
    position_ids_full = torch.stack(position_ids_full_list, dim=0)  # [B, seq_len + K]
    attention_mask_expanded = torch.stack(attention_mask_expanded_list, dim=0)  # [B, seq_len + K]

    dummy_input_ids = torch.zeros(batch_size, expanded_seq_len, dtype=torch.long, device=device)

    logits, final_hidden, intervened_hidden = run_remaining_forward(
        model, H_expanded, attention_mask_expanded, position_ids_full,
        start_from_layer=extract_layer + 1, input_ids=dummy_input_ids,
        extract_layer=extract_layer  
    )

    ce_losses = []
    for b in range(batch_size):
        M = int(M_list[b])

        response_start_in_labels = M

        response_labels_b = labels[b, response_start_in_labels:]
        valid_mask = (response_labels_b != -100)
        
        if valid_mask.any():
            response_len_b = valid_mask.sum().item()
            
            response_logits_b = logits[b, M + K - 1 : M + K - 1 + response_len_b, :]  # [response_len, V]
            response_labels_valid = response_labels_b[valid_mask]  # [response_len]
            
            loss_b = F.cross_entropy(response_logits_b, response_labels_valid)
            ce_losses.append(loss_b)
    
    if len(ce_losses) > 0:
        loss_ce = torch.stack(ce_losses).mean()
    else:
        loss_ce = torch.tensor(0.0, device=device, requires_grad=True)
    
    safety_scores = []
    for b in range(batch_size):
        M = int(M_list[b])
        
        prompt_hidden_b = intervened_hidden[b, :M, :]  # [M, D]
        prompt_mask_b = torch.ones(M, dtype=torch.long, device=device)

        response_hidden_b = intervened_hidden[b, M + K:, :]  # [L, D]
        response_mask_b = attention_mask[b, M:]  # [L] 原始 response mask

        valid_indices = response_mask_b.bool()
        if valid_indices.any() and prompt_hidden_b.size(0) > 0:
            response_hidden_valid = response_hidden_b[valid_indices]  # [valid_len, D]
            if response_hidden_valid.size(0) > 0:
                resp_attn_mask = torch.ones(response_hidden_valid.size(0), dtype=torch.long, device=device)

                value_repr = model.value_transformer.forward_conditional(
                    prompt_hidden_b.unsqueeze(0),      # [1, M, D]
                    response_hidden_valid.unsqueeze(0), # [1, valid_len, D]
                    prompt_mask_b.unsqueeze(0),        # [1, M]
                    resp_attn_mask.unsqueeze(0)        # [1, valid_len]
                )
                safety_score = model.discriminator(value_repr)
                safety_scores.append(safety_score)
    
    if len(safety_scores) > 0:
        scores = torch.cat(safety_scores, dim=0)
        alpha = 2.0 
        loss_safe = (F.softplus(scores) + alpha * F.relu(scores)).mean()
    else:
        loss_safe = torch.tensor(0.0, device=device)
    
    prompt_norms = []
    for b in range(batch_size):
        M = int(M_list[b])
        if M > 0:
            H_prompt_b = hidden_states_at_extract[b, :M, :]
            prompt_norm = torch.norm(H_prompt_b, dim=-1).mean()
            prompt_norms.append(prompt_norm)
    
    if len(prompt_norms) > 0:
        target_norm = torch.stack(prompt_norms).mean() * 0.1
    else:
        target_norm = torch.tensor(1.0, device=device)
    
    if is_vlp:
        trigger_norm = torch.norm(trigger_hidden, dim=-1).mean() 
        bridge_norm = torch.norm(value_bridge, dim=-1).mean()  

        ratio = bridge_norm / (trigger_norm + 1e-8)
        loss_norm = torch.clamp(torch.abs(ratio - 1.0) - 0.2, min=0.0) 
    else:
        actual_norm = torch.norm(value_adjustment, dim=-1).mean()
        loss_norm = torch.clamp(actual_norm - target_norm, min=0.0)
    
    total_loss = lambda_ce * loss_ce + lambda_safe * loss_safe + lambda_reg * loss_norm
    
    return {
        'loss_ce': loss_ce,
        'loss_safe': loss_safe,
        'loss_reg': loss_norm,
        'total_loss': total_loss,
        'generated_tokens': value_bridge.detach(),
    }


def run_partial_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    stop_at_layer: int,
    requires_grad: bool = False,
) -> torch.Tensor:

    forward_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'output_hidden_states': True,
    }
    
    if not requires_grad:
        with torch.no_grad():
            outputs = model.base_model(**forward_kwargs)
    else:
        outputs = model.base_model(**forward_kwargs)
    
    # hidden_states[0] = embeddings, hidden_states[N+1] = layer N output
    return outputs.hidden_states[stop_at_layer + 1]


def run_remaining_forward(
    model,
    modified_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    start_from_layer: int,
    input_ids: torch.Tensor = None,
    extract_layer: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if input_ids is None:
        raise ValueError("input_ids is required for run_remaining_forward")
    
    batch_size, seq_len, hidden_dim = modified_hidden.shape
    assert input_ids.shape[1] == seq_len, f"input_ids length {input_ids.shape[1]} != modified_hidden length {seq_len}"
    assert attention_mask.shape[1] == seq_len, f"attention_mask length {attention_mask.shape[1]} != modified_hidden length {seq_len}"
    assert position_ids.shape[1] == seq_len, f"position_ids length {position_ids.shape[1]} != modified_hidden length {seq_len}"
    
    if hasattr(model.base_model, 'transformer'):
        # GPT-2 style
        layers = model.base_model.transformer.h
    elif hasattr(model.base_model, 'model'):
        # Llama/Mistral/Qwen style
        layers = model.base_model.model.layers
    else:
        raise ValueError(f"Unsupported model architecture: {type(model.base_model)}")

    hook_layer_idx = start_from_layer - 1
    hook_layer = layers[hook_layer_idx]
    
    replacement_hidden = {'value': modified_hidden}
    
    def replacement_hook(module, input, output):
        if isinstance(output, tuple):
            return (replacement_hidden['value'],) + output[1:]
        else:
            return replacement_hidden['value']
    
    hook_handle = hook_layer.register_forward_hook(replacement_hook)
    
    try:
        outputs = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits
        final_hidden = outputs.hidden_states[-1]
    
        if extract_layer is not None:
            target_layer_idx = min(extract_layer + 2, len(outputs.hidden_states) - 1)
            intervened_hidden = outputs.hidden_states[target_layer_idx]
        else:
            intervened_hidden = outputs.hidden_states[start_from_layer + 1]
    finally:
        hook_handle.remove()
    
    return logits, final_hidden, intervened_hidden


def train_epoch_detailed(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    n_epochs,
    use_gradient_delta: bool = True,
    gradient_step_size: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_safe: float = 0.5,
    lambda_reg: float = 0.1,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
):
    model.train()
    train_loss = 0.0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        prompt_len = batch['prompt_len'].to(device)
            
        loss_dict = compute_losses_detailed_sft(
            model=model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prompt_len=prompt_len,
            device=device,
            use_gradient_delta=use_gradient_delta,
            gradient_step_size=gradient_step_size,
            lambda_ce=lambda_ce,
            lambda_safe=lambda_safe,
            lambda_reg=lambda_reg,
        )
        
        loss = loss_dict['total_loss']
        
        if not loss.requires_grad:
            raise RuntimeError("total_loss does not require grad! Check gradient flow.")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        train_loss += loss.item()
        batch_size = input_ids.size(0)
        train_total += batch_size

        if hasattr(model.generator, 'gate_alpha'):
            gating_factor = F.softplus(model.generator.gate_alpha).item()
        elif hasattr(model.generator, 'gating_alpha'):
            gating_factor = F.softplus(model.generator.gating_alpha).item()
        else:
            gating_factor = 0.0
            
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{loss_dict["loss_ce"].item():.4f}',
            'safe': f'{loss_dict["loss_safe"].item():.4f}',
            'reg': f'{loss_dict["loss_reg"].item():.4f}',
            'gate': f'{gating_factor:.4f}',
        })
    
    return train_loss / len(train_loader)


def validate_epoch(
    model, 
    val_loader, 
    device,
    use_gradient_delta: bool = True,
    gradient_step_size: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_safe: float = 0.5,
    lambda_reg: float = 0.1,
):
    model.eval()
    val_loss = 0.0
    
    metrics_sum = {
        'loss_ce': 0.0,
        'loss_safe': 0.0,
        'loss_reg': 0.0,
    }
    
    num_batches = 0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        prompt_len = batch['prompt_len'].to(device)
            
        grad_context = torch.enable_grad() if use_gradient_delta else torch.no_grad()
        with grad_context:
            loss_dict = compute_losses_detailed_sft(
                model=model,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_len=prompt_len,
                device=device,
                use_gradient_delta=use_gradient_delta,
                gradient_step_size=gradient_step_size,
                lambda_ce=lambda_ce,
                lambda_safe=lambda_safe,
                lambda_reg=lambda_reg,
            )
            loss_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in loss_dict.items()}
            
        metrics_sum['loss_ce'] += loss_dict['loss_ce'].item()
        metrics_sum['loss_safe'] += loss_dict['loss_safe'].item()
        metrics_sum['loss_reg'] += loss_dict['loss_reg'].item()
        
        val_loss += loss_dict['total_loss'].item()
        num_batches += 1
    
    avg_val_loss = val_loss / num_batches
    avg_metrics = {
        'loss_ce': metrics_sum['loss_ce'] / num_batches,
        'loss_safe': metrics_sum['loss_safe'] / num_batches,
        'loss_reg': metrics_sum['loss_reg'] / num_batches,
    }
    
    return avg_val_loss, avg_metrics


def train_stage3(
    config_path: str,
    stage2_checkpoint: str = None,
    device: str = "cuda",
    subset_size: int = None,
    **overrides,
):
    """
    Train Stage 3 using configuration file
    
    Args:
        config_path: Path to YAML configuration file
        stage2_checkpoint: Path to Stage 2 checkpoint (if None, auto-detect from config)
        device: Device to use
        subset_size: Optional subset size for debugging
        **overrides: Command-line arguments to override config
    """
    # Load configuration
    config = load_config(config_path, **overrides)
    
    # Extract training config
    train_config = config.get('training', {}).get('stage3', {})
    paths_config = config.get('paths', {})
    arch_config = config.get('architecture', {})
    gen_config = config.get('generator', {})
    
    # Auto-detect stage2 checkpoint if not provided
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if stage2_checkpoint is None:
        checkpoint_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
        stage2_checkpoint = os.path.join(checkpoint_dir, 'stage2_best.pt')
    else:
        # Resolve relative paths
        if not os.path.isabs(stage2_checkpoint):
            stage2_checkpoint = os.path.join(svgt_dir, stage2_checkpoint)
    
    # Create model from config
    model = BaseValueModel(config, device=device)
    
    # Load Stage 2 checkpoint
    if not os.path.exists(stage2_checkpoint):
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {stage2_checkpoint}\n"
            f"Please train Stage 2 first or provide a valid checkpoint path."
        )
    
    print(f"Loading stage 2 checkpoint: {stage2_checkpoint}")
    try:
        checkpoint = torch.load(stage2_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.value_transformer.load_state_dict(checkpoint['model_state_dict']['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
        else:
            model.value_transformer.load_state_dict(checkpoint['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['discriminator'])
        
        print("Stage 2 weights loaded")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Stage 2 checkpoint from {stage2_checkpoint}: {e}\n"
            f"Please check if the checkpoint file is corrupted or in the correct format."
        )
    
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    model.freeze_for_stage3()
    
    # Architecture detection and logging
    arch_type, num_layers = detect_architecture(model)
    print(f"\n🔍 Detected Architecture: {arch_type}")
    print(f"  - Total layers: {num_layers}")
    print(f"  - Extract layer: {model.extract_layer}")
    print(f"  - Intervention tokens: {model.n_intervention_tokens}")
    if model.extract_layer >= num_layers:
        raise ValueError(
            f"Extract layer ({model.extract_layer}) must be less than total layers ({num_layers})"
        )
    
    print("\nChecking generator parameters:")
    generator_params = list(model.generator.parameters())
    trainable_params = [p for p in generator_params if p.requires_grad]
    total_param_count = sum(p.numel() for p in generator_params)
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"  Total generator parameter tensors: {len(generator_params)}")
    print(f"  Trainable generator parameter tensors: {len(trainable_params)}")
    print(f"  Total generator parameters (elements): {total_param_count:,}")
    print(f"  Trainable generator parameters (elements): {trainable_param_count:,}")
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters in generator! Check freeze_for_stage3().")
    
    # Data loading - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(svgt_dir, paths_config.get('data_dir', 'data/processed'), 'stage2')
    tokenizer = model.tokenizer
    # Load paired data (safe + harmful responses for each prompt)
    train_data = load_paired_data(os.path.join(data_dir, "train.json"))
    val_data = load_paired_data(os.path.join(data_dir, "val.json"))
    
    if subset_size is not None:
        train_data = train_data[:subset_size]
        val_data = val_data[:min(subset_size, 1000)]
    
    print(f"\nData statistics:")
    print(f"  Training set: {len(train_data)} safe samples")
    print(f"  Validation set: {len(val_data)} safe samples")
    
    # Save temporary files to data_dir instead of system temp directory
    os.makedirs(data_dir, exist_ok=True)
    train_temp_path = os.path.join(data_dir, 'train_stage3_temp.json')
    val_temp_path = os.path.join(data_dir, 'val_stage3_temp.json')
    
    with open(train_temp_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    
    with open(val_temp_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f)
    
    # Use SFT format dataset
    n_value_tokens = arch_config.get('n_intervention_tokens', 1)
    train_dataset = Stage3Dataset(
        train_temp_path, 
        tokenizer, 
        n_value_tokens=n_value_tokens,
        max_length=512
    )
    val_dataset = Stage3Dataset(
        val_temp_path, 
        tokenizer, 
        n_value_tokens=n_value_tokens,
        max_length=512
    )
    
    # Extract training parameters from config
    use_gradient_delta = train_config.get('use_gradient_delta', True)
    gradient_step_size = train_config.get('gradient_step_size', 1.0)
    lambda_ce = train_config.get('lambda_ce', 1.0)  # Language model loss weight
    lambda_safe = train_config.get('lambda_safe', 0.5)  # Safety loss weight
    lambda_reg = train_config.get('lambda_reg', 0.1)  # Regularization loss weight
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    log_interval = train_config.get('log_interval', 100)
    batch_size = train_config.get('batch_size', 4)
    n_epochs = train_config.get('n_epochs', 5)
    lr_new = train_config.get('lr_new', 5e-4)
    lr_finetune = train_config.get('lr_finetune', 1e-5)
    value_dim = arch_config.get('value_dim', 128)
    
    # Using gradient method (default and recommended)
    print("\nUsing gradient method for value token generation.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    parameter_groups = model.get_stage3_parameter_groups(lr_new=lr_new, lr_finetune=lr_finetune)
    
    print("\nParameter groups:")
    for i, group in enumerate(parameter_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i+1} ({group['name']}): lr={group['lr']}, num_params={num_params:,}")
    
    optimizer = optim.AdamW(parameter_groups, weight_decay=1e-5)
    
    # Save directory - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # Training metrics tracking
    train_metrics_history = {
        'loss': [],
        'loss_ce': [],
        'loss_safe': [],
        'loss_reg': [],
    }
    val_metrics_history = {
        'loss': [],
        'loss_ce': [],
        'loss_safe': [],
        'loss_reg': [],
    }
    
    for epoch in range(n_epochs):
        # Training
        avg_train_loss = train_epoch_detailed(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            n_epochs=n_epochs,
            use_gradient_delta=use_gradient_delta,
            gradient_step_size=gradient_step_size,
            lambda_ce=lambda_ce,
            lambda_safe=lambda_safe,
            lambda_reg=lambda_reg,
            max_grad_norm=max_grad_norm,
            log_interval=log_interval,
        )
        
        # Validation
        avg_val_loss, val_metrics = validate_epoch(
            model, 
            val_loader, 
            device,
            use_gradient_delta=use_gradient_delta,
            gradient_step_size=gradient_step_size,
            lambda_ce=lambda_ce,
            lambda_safe=lambda_safe,
            lambda_reg=lambda_reg,
        )
        
        # Record metrics
        with torch.no_grad():
            train_metrics_history['loss'].append(avg_train_loss)
            val_metrics_history['loss'].append(avg_val_loss)
            val_metrics_history['loss_ce'].append(val_metrics['loss_ce'])
            val_metrics_history['loss_safe'].append(val_metrics['loss_safe'])
            val_metrics_history['loss_reg'].append(val_metrics['loss_reg'])
        
        if hasattr(model.generator, 'gate_alpha'):
            gating_factor = F.softplus(model.generator.gate_alpha).item()
        elif hasattr(model.generator, 'gating_alpha'):
            gating_factor = F.softplus(model.generator.gating_alpha).item()
        else:
            gating_factor = 0.0
        
        print(f"\nEpoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  CE Loss: {val_metrics_history['loss_ce'][-1]:.4f}")
        print(f"  Safe Loss: {val_metrics_history['loss_safe'][-1]:.4f}")
        print(f"  Reg Loss: {val_metrics_history['loss_reg'][-1]:.4f}")
        print(f"  Gating Factor: {gating_factor:.4f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': {
                'value_transformer': model.value_transformer.state_dict(),
                'discriminator': model.discriminator.state_dict(),
                'generator': model.generator.state_dict(),
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'training_config': {
                'use_gradient_delta': use_gradient_delta,
                'gradient_step_size': gradient_step_size,
                'use_transformer_projector': gen_config.get('use_transformer_projector', False),
                'lambda_ce': lambda_ce,
                'lambda_safe': lambda_safe,
                'lambda_reg': lambda_reg,
            },
            'config': config,  # Save full config for reproducibility
        }
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'stage3_best.pt'))
            print(f"  Saved best checkpoint (val_loss: {avg_val_loss:.4f})")
        
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'stage3_epoch_{epoch+1}.pt'))

    # Clean up temporary files if they exist
    if os.path.exists(train_temp_path):
        os.unlink(train_temp_path)
    if os.path.exists(val_temp_path):
        os.unlink(val_temp_path)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"\nStage 3 training completed:")
    generator_type = "TransformerValueProjector" if gen_config.get('use_transformer_projector', False) else "TokenGenerator"
    method_type = "gradient method" if use_gradient_delta else "simple subtraction method"
    print(f"   - Using SFT format with dense supervision")
    print(f"   - Using {generator_type} to generate intervention tokens")
    print(f"   - Using {method_type} for value correction")
    print(f"   - Loss weights: CE={lambda_ce:.1f}, Safe={lambda_safe:.1f}, Reg={lambda_reg:.1f}")
    print(f"   - Value tokens: {n_value_tokens}, Extract layer: {model.extract_layer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Training with Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--stage2_checkpoint", type=str, default=None, help="Path to Stage 2 checkpoint (auto-detect if not provided)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for debugging")
    
    args, unknown = parser.parse_known_args()
    
    overrides = parse_cli_overrides(unknown)
    
    train_stage3(
        config_path=args.config,
        stage2_checkpoint=args.stage2_checkpoint,
        device=args.device,
        subset_size=args.subset_size,
        **overrides,
    )
