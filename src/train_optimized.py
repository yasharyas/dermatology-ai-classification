"""
Training script optimized for RTX 3060 Laptop GPU (6GB VRAM)
- Lightweight augmentations for fast CPU processing
- Optimal batch size for 6GB VRAM
- Gradient accumulation for effective larger batches
- Mixed precision (FP16) for 2x speed on Ampere
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import HAM10000Dataset
from models.model import create_model
from training.losses import FocalLoss
from training.metrics import evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnetv2_rw_m',
                       choices=['efficientnetv2_rw_m', 'convnext_large', 'swin_large', 'vit_large'])
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)  # Per-GPU batch, will accumulate
    parser.add_argument('--accumulation_steps', type=int, default=2)  # Effective batch = 64
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=6)  # More workers for CPU augmentations
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_every', type=int, default=10)  # Save checkpoint every N epochs
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Train one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast('cuda'):
            outputs = model(images, None)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Stats
        running_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return running_loss/len(loader), correct/total


def validate(model, loader, criterion, device):
    """Validate"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with autocast('cuda'):
                outputs = model(images, None)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}'})
    
    # Calculate metrics
    metrics = evaluate(
        np.array(all_labels),
        np.array(all_preds),
        np.eye(7)[all_preds]  # one-hot for predictions
    )
    
    return running_loss/len(loader), metrics


def main():
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 RTX 3060 OPTIMIZED TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"CUDA Cores: {props.multi_processor_count * 128}")
    
    # Load dataset stats
    stats_file = Path('d:/CODING/dataverse_files/processed/dataset_stats.json')
    with open(stats_file) as f:
        stats = json.load(f)
    
    effective_batch = args.batch_size * args.accumulation_steps
    print(f"\n📊 Dataset: {stats['total_samples']} samples, {stats['num_classes']} classes")
    print(f"   Batch size: {args.batch_size} (accumulation: {args.accumulation_steps}, effective: {effective_batch})")
    
    # Create model
    print(f"\n📦 Creating {args.model}...")
    model_config = {
        'model_name': args.model,
        'num_classes': 7,
        'pretrained': True,
        'use_metadata': False,
        'dropout': 0.3
    }
    model = create_model(model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    
    # Create dataloaders with lightweight augmentations
    print("\n📚 Loading data...")
    train_dataset = HAM10000Dataset(
        csv_file='d:/CODING/dataverse_files/processed/train.csv',
        transform=HAM10000Dataset.get_train_transforms(args.img_size),
        use_metadata=False,
        img_size=args.img_size
    )
    
    val_dataset = HAM10000Dataset(
        csv_file='d:/CODING/dataverse_files/processed/val.csv',
        transform=HAM10000Dataset.get_val_transforms(args.img_size),
        use_metadata=False,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers == 0 else True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers == 0 else True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Loss and optimizer
    class_weights = torch.tensor(stats['train_class_weights']).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Differential learning rates: lower for backbone, higher for head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # 10x lower for backbone
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-4)
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed precision scaler (optimized for Ampere)
    scaler = GradScaler('cuda')
    
    # Output directory
    output_dir = Path(args.output_dir) / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*80}\n")
    
    best_acc = 0.0
    best_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.accumulation_steps
        )
        train_time = time.time() - epoch_start
        
        # Validate
        val_start = time.time()
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_time = time.time() - val_start
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        # Print stats
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Time: {train_time:.1f}s")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Time: {val_time:.1f}s")
        print(f"   Val F1: {val_metrics['f1_macro']:.4f} | Precision: {val_metrics['precision_macro']:.4f} | Recall: {val_metrics['recall_macro']:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_f1 = val_metrics['f1_macro']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'accuracy': best_acc,
                'f1_score': best_f1,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"   ✅ Saved best model (Acc: {best_acc*100:.2f}%, F1: {best_f1:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'accuracy': val_metrics['accuracy'],
                'f1_score': val_metrics['f1_macro'],
                'history': history
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch+1}.pth')
            print(f"   💾 Saved checkpoint: epoch {epoch+1}")
    
    total_time = time.time() - total_start
    
    # Save final history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
