"""
Stage 2: Conditional Value Learning
Unified training script using configuration system
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    from SVGT.training.stage2_loader import Stage2Dataset
    from SVGT.models import BaseValueModel
    from SVGT.utils import load_config, parse_cli_overrides
except ModuleNotFoundError:
    from training.stage2_loader import Stage2Dataset
    from models import BaseValueModel
    from utils import load_config, parse_cli_overrides


def train_stage2(
    config_path: str,
    stage1_checkpoint: str = None,
    device: str = "cuda",
    subset_size: int = None,
    **overrides,
):
    """
    Train Stage 2 using configuration file
    
    Args:
        config_path: Path to YAML configuration file
        stage1_checkpoint: Path to Stage 1 checkpoint (if None, auto-detect from config)
        device: Device to use
        subset_size: Optional subset size for debugging
        **overrides: Command-line arguments to override config
    """
    # Load configuration
    config = load_config(config_path, **overrides)
    
    # Extract training config
    train_config = config.get('training', {}).get('stage2', {})
    paths_config = config.get('paths', {})
    
    # Auto-detect stage1 checkpoint if not provided
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if stage1_checkpoint is None:
        checkpoint_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
        stage1_checkpoint = os.path.join(checkpoint_dir, 'stage1_best.pt')
    else:
        # Resolve relative paths
        if not os.path.isabs(stage1_checkpoint):
            stage1_checkpoint = os.path.join(svgt_dir, stage1_checkpoint)
    
    # Create model from config
    model = BaseValueModel(config, device=device)
    
    # Load Stage 1 checkpoint
    if not os.path.exists(stage1_checkpoint):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {stage1_checkpoint}\n"
            f"Please train Stage 1 first or provide a valid checkpoint path."
        )
    
    print(f"Loading stage 1 checkpoint: {stage1_checkpoint}")
    try:
        checkpoint = torch.load(stage1_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.value_transformer.load_stage1_weights(checkpoint['model_state_dict']['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
        else:
            model.value_transformer.load_stage1_weights(checkpoint['value_transformer'])
            model.discriminator.load_state_dict(checkpoint['discriminator'])
        
        print("Load stage 1 weights successfully!")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Stage 1 checkpoint from {stage1_checkpoint}: {e}\n"
            f"Please check if the checkpoint file is corrupted or in the correct format."
        )
    
    model.freeze_for_stage2()
    
    # Data loading - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(svgt_dir, paths_config.get('data_dir', 'data/processed'), 'stage2')
    tokenizer = model.tokenizer
    
    train_dataset = Stage2Dataset(
        os.path.join(data_dir, "train.json"),
        tokenizer,
        max_length=512,
        subset_size=subset_size,
    )
    val_dataset = Stage2Dataset(
        os.path.join(data_dir, "val.json"),
        tokenizer,
        max_length=512,
        subset_size=min(subset_size or 1000, 1000) if subset_size else 1000,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Optimizer with parameter groups
    parameter_groups = model.get_stage2_parameter_groups(
        lr_new=train_config.get('lr_new', 5e-4),
        lr_finetune=train_config.get('lr_finetune', 1e-5),
    )
    
    print("\nParams Groups:")
    for i, group in enumerate(parameter_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Groups {i+1} ({group['name']}): lr={group['lr']}, num_params={num_params:,}")
    
    optimizer = optim.AdamW(parameter_groups, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Save directory - resolve paths relative to SVGT directory
    svgt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(svgt_dir, paths_config.get('checkpoint_dir', 'checkpoints/default'))
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    n_epochs = train_config.get('n_epochs', 5)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            prompt_ids = batch['prompt_ids'].to(device)
            response_ids = batch['response_ids'].to(device)
            prompt_mask = batch['prompt_mask'].to(device)
            response_mask = batch['response_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            scores = model.forward_stage2(
                prompt_ids, response_ids,
                prompt_mask, response_mask
            )
            
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.value_transformer.parameters()) + list(model.discriminator.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(scores) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
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
                prompt_ids = batch['prompt_ids'].to(device)
                response_ids = batch['response_ids'].to(device)
                prompt_mask = batch['prompt_mask'].to(device)
                response_mask = batch['response_mask'].to(device)
                labels = batch['labels'].to(device)
                
                scores = model.forward_stage2(
                    prompt_ids, response_ids,
                    prompt_mask, response_mask
                )
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
                'config': config,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'stage2_best.pt'))
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
            torch.save(checkpoint, os.path.join(save_dir, f'stage2_epoch_{epoch+1}.pt'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training with Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Path to Stage 1 checkpoint (auto-detect if not provided)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for debugging")
    
    args, unknown = parser.parse_known_args()
    
    overrides = parse_cli_overrides(unknown)
    
    train_stage2(
        config_path=args.config,
        stage1_checkpoint=args.stage1_checkpoint,
        device=args.device,
        subset_size=args.subset_size,
        **overrides,
    )
