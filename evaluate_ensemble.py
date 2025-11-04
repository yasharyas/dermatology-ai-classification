"""
Ensemble inference script combining multiple trained models
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.dataset import HAM10000Dataset
from torch.utils.data import DataLoader
from models.model import create_model
from tqdm import tqdm

class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of model instances
            weights: List of weights for each model (optional)
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.weights = np.array(self.weights) / sum(self.weights)  # Normalize
        
    def predict(self, x):
        """
        Ensemble prediction by weighted averaging
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            predictions: Averaged logits [B, num_classes]
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output = model(x, None)
                    predictions.append(output * weight)
        
        # Average predictions
        ensemble_output = torch.stack(predictions).sum(dim=0)
        return ensemble_output


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = {
        'model_name': ckpt.get('model_name', 'efficientnetv2_rw_s'),
        'num_classes': 7,
        'pretrained': False,  # Already trained
        'use_metadata': False,
        'dropout': 0.2
    }
    
    model = create_model(model_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"   Model: {model_config['model_name']}")
    print(f"   Val Acc: {ckpt['val_acc']:.2f}%")
    
    return model, ckpt['val_acc']


def evaluate_ensemble(model_paths, val_csv='processed/val.csv', img_size=256, device='cuda'):
    """Evaluate ensemble on validation set"""
    
    print("="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    
    # Load all models
    models = []
    weights = []
    
    for path in model_paths:
        if Path(path).exists():
            model, val_acc = load_model_from_checkpoint(path, device)
            models.append(model)
            weights.append(val_acc)  # Weight by validation accuracy
        else:
            print(f"⚠️  Model not found: {path}")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"\n📊 Loaded {len(models)} models")
    print(f"   Weights (by val acc): {[f'{w:.2f}%' for w in weights]}")
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights)
    
    # Create validation dataset
    print(f"\nLoading validation dataset...")
    val_dataset = HAM10000Dataset(
        csv_file=val_csv,
        transform=HAM10000Dataset.get_val_transforms(img_size),
        use_metadata=False,
        img_size=img_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Val samples: {len(val_dataset)}")
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    print("\nEvaluating ensemble...")
    for batch in tqdm(val_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Ensemble prediction
        outputs = ensemble.predict(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    
    print("\n" + "="*80)
    print(f"ENSEMBLE RESULTS")
    print("="*80)
    print(f"Number of models: {len(models)}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Individual model accuracies
    print(f"\nIndividual model accuracies:")
    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        if Path(path).exists():
            print(f"  Model {i+1}: {weight:.2f}%")
    
    improvement = accuracy - max(weights) if weights else 0
    print(f"\nEnsemble improvement: +{improvement:.2f}%")
    print("="*80)
    
    return accuracy


if __name__ == '__main__':
    # Model checkpoints
    model_paths = [
        'checkpoints/best_model_fast.pth',  # EfficientNetV2-RW-S
        'checkpoints/best_model_convnext.pth',  # ConvNeXt-Tiny
    ]
    
    # Evaluate ensemble
    evaluate_ensemble(model_paths, img_size=256)
