"""
Stage 1: Unconditional Value Learning
Unified training script using configuration system
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
package_parent = os.path.dirname(repo_root)
for path in (package_parent, repo_root):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from SVGT.training.stage1_loader import Stage1Dataset
    from SVGT.models import BaseValueModel
    from SVGT.utils import load_config, parse_cli_overrides
except ModuleNotFoundError:
    from training.stage1_loader import Stage1Dataset
    from models import BaseValueModel
    from utils import load_config, parse_cli_overrides


def train_stage1(
    config_path: str,
    device: str = "cuda",
    subset_size: int = None,
    **overrides,
):
    """
    Train Stage 1 using configuration file
    
    Args:
        config_path: Path to YAML configuration file
        device: Device to use
        subset_size: Optional subset size for debugging
        **overrides: Command-line arguments to override config (e.g., training.stage1.batch_size=16)
    """
    # Load configuration
    config = load_config(config_path, **overrides)
    
    # Extract training config
    train_config = config.get('training', {}).get('stage1', {})
    paths_config = config.get('paths', {})
    
    # Create model from config
    model = BaseValueModel(config, device=device)
    model.freeze_for_stage1()
    
    # Data loading - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(svgt_dir, paths_config.get('data_dir', 'data/processed'), 'stage1')
    tokenizer = model.tokenizer
    
    train_dataset = Stage1Dataset(
        os.path.join(data_dir, "train.json"),
        tokenizer,
        max_length=512,
        subset_size=subset_size,
    )
    val_dataset = Stage1Dataset(
        os.path.join(data_dir, "val.json"),
        tokenizer,
        max_length=512,
        subset_size=min(subset_size or 1000, 1000) if subset_size else 1000,
    )
    
    # Get num_workers from config or use default (4 for multi-core, 0 for single-core)
    num_workers = train_config.get('num_workers', 4)
    # Use 0 if on Windows or if explicitly set to 0
    if num_workers > 0 and sys.platform == 'win32':
        num_workers = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.startswith('cuda') else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.startswith('cuda') else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        list(model.value_transformer.parameters()) + list(model.discriminator.parameters()),
        lr=train_config.get('lr', 1e-4),
        weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision training (AMP) - significantly faster
    use_amp = train_config.get('use_amp', True)
    scaler = GradScaler() if use_amp and device.startswith('cuda') else None
    
    # Save directory - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    n_epochs = train_config.get('n_epochs', 10)
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
    
            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with autocast():
                    scores = model.forward_stage1(input_ids, attention_mask)
                    loss = criterion(scores, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                scores = model.forward_stage1(input_ids, attention_mask)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(scores) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            loss_str = f'{loss.item():.4f}' if not (torch.isnan(loss) or torch.isinf(loss)) else 'nan'
            pbar.set_postfix({
                'loss': loss_str,
                'acc': f'{train_correct/train_total:.4f}',
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Mixed precision for validation too
                if use_amp and scaler is not None:
                    with autocast():
                        scores = model.forward_stage1(input_ids, attention_mask)
                        loss = criterion(scores, labels)
                else:
                    scores = model.forward_stage1(input_ids, attention_mask)
                    loss = criterion(scores, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(scores) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': {
                    'value_transformer': model.value_transformer.state_dict(),
                    'discriminator': model.discriminator.state_dict(),
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'config': config,  # Save config for reproducibility
            }
            torch.save(checkpoint, os.path.join(save_dir, 'stage1_best.pt'))
            print(f"Saved best checkpoint (val_acc: {val_acc:.4f})")
        
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': {
                    'value_transformer': model.value_transformer.state_dict(),
                    'discriminator': model.discriminator.state_dict(),
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'stage1_epoch_{epoch+1}.pt'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Training with Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for debugging")
    
    # Allow overriding any config parameter with dot notation
    # Example: --training.stage1.batch_size 16
    args, unknown = parser.parse_known_args()
    
    overrides = parse_cli_overrides(unknown)
    
    train_stage1(
        config_path=args.config,
        device=args.device,
        subset_size=args.subset_size,
        **overrides,
    )
