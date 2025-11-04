"""
Ultra-fast training script with aggressive optimizations
- Smaller model (EfficientNetV2-RW-S: 24M params vs 51M)
- Smaller image size (256px vs 384px)
- Larger batch size
- Compile model (PyTorch 2.0+ speedup)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.dataset import HAM10000Dataset
from models.model import create_model
from training.losses import FocalLoss
from training.metrics import evaluate
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnetv2_rw_s', 
                       choices=['efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'convnext_tiny', 'convnext_small'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for speedup')
    parser.add_argument('--output', type=str, default='best_model_fast.pth', help='Output checkpoint name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("ULTRA-FAST TRAINING MODE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Image Size: {args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Compile: {args.compile}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create datasets with lightweight augmentations
    print("\nLoading datasets...")
    train_dataset = HAM10000Dataset(
        csv_file='processed/train.csv',
        transform=HAM10000Dataset.get_train_transforms(args.img_size, heavy_aug=False),
        use_metadata=False,
        img_size=args.img_size
    )
    
    val_dataset = HAM10000Dataset(
        csv_file='processed/val.csv',
        transform=HAM10000Dataset.get_val_transforms(args.img_size),
        use_metadata=False,
        img_size=args.img_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model_config = {
        'model_name': args.model,
        'num_classes': 7,
        'pretrained': True,
        'use_metadata': False,
        'dropout': 0.2
    }
    model = create_model(model_config).to(device)
    
    # Compile model for speedup (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Setup training
    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images, None)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            acc = (np.array(train_preds) == np.array(train_labels)).mean() * 100
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        train_acc = (np.array(train_preds) == np.array(train_labels)).mean() * 100
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images, None)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean() * 100
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': args.model,
            }, f'checkpoints/{args.output}')
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print("="*80)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
