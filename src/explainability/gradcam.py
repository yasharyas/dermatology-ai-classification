"""
Grad-CAM and explainability visualizations for skin lesion classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Paper: https://arxiv.org/abs/1610.02391
    
    Highlights which regions of the image are important for the model's decision
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained model
            target_layer: Layer to compute gradients (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class for visualization (if None, use predicted class)
        
        Returns:
            heatmap: Grad-CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H', W']
        activations = self.activations[0]  # [C, H', W']
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    @staticmethod
    def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
        """
        Apply colormap to heatmap
        
        Args:
            heatmap: Grayscale heatmap [H, W]
            colormap: OpenCV colormap
        
        Returns:
            colored_heatmap: RGB heatmap [H, W, 3]
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        return colored_heatmap
    
    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image [H, W, 3] (0-255)
            heatmap: Heatmap [H, W] (0-1)
            alpha: Overlay transparency
        
        Returns:
            overlayed: Overlayed image [H, W, 3]
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        colored_heatmap = GradCAM.apply_colormap(heatmap_resized)
        
        # Overlay
        overlayed = (alpha * image + (1 - alpha) * colored_heatmap).astype(np.uint8)
        
        return overlayed


def visualize_gradcam(
    model,
    image_path,
    transform,
    class_names,
    device='cuda',
    save_path=None
):
    """
    Generate and visualize Grad-CAM for an image
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image preprocessing transform
        class_names: List of class names
        device: Device to run on
        save_path: Path to save visualization
    """
    # Load and preprocess image
    image_orig = cv2.imread(str(image_path))
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    
    # Transform
    image_tensor = transform(image=image_orig)['image'].unsqueeze(0).to(device)
    
    # Find target layer (last conv layer before global pooling)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        print("⚠️  Could not find convolutional layer for Grad-CAM")
        return
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    with torch.set_grad_enabled(True):
        heatmap = gradcam.generate(image_tensor)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[pred_class].item()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_orig)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    overlayed = GradCAM.overlay_heatmap(image_orig, heatmap, alpha=0.5)
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay\nPrediction: {class_names[pred_class]} ({pred_prob*100:.2f}%)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Grad-CAM visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return heatmap, pred_class, pred_prob


def visualize_multiple_classes(
    model,
    image_path,
    transform,
    class_names,
    device='cuda',
    save_path=None
):
    """
    Visualize Grad-CAM for all classes
    
    Shows which regions activate for each possible diagnosis
    """
    # Load image
    image_orig = cv2.imread(str(image_path))
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image=image_orig)['image'].unsqueeze(0).to(device)
    
    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap for each class
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, (num_classes + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(num_classes):
        with torch.set_grad_enabled(True):
            heatmap = gradcam.generate(image_tensor, target_class=i)
        
        overlayed = GradCAM.overlay_heatmap(image_orig, heatmap, alpha=0.4)
        axes[i].imshow(overlayed)
        axes[i].set_title(f'{class_names[i]}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('Grad-CAM for All Classes', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved multi-class visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def batch_gradcam_visualization(
    model,
    dataloader,
    class_names,
    output_dir,
    num_samples=20,
    device='cuda'
):
    """
    Generate Grad-CAM visualizations for a batch of images
    
    Args:
        model: Trained model
        dataloader: DataLoader
        class_names: List of class names
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        device: Device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    gradcam = GradCAM(model, target_layer)
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        
        images = batch['image'].to(device)
        labels = batch['label'].cpu().numpy()
        image_ids = batch['image_id']
        
        for i in range(len(images)):
            if count >= num_samples:
                break
            
            # Generate heatmap
            with torch.set_grad_enabled(True):
                heatmap = gradcam.generate(images[i:i+1])
            
            # Get prediction
            with torch.no_grad():
                output = model(images[i:i+1])
                pred_class = output.argmax(dim=1).item()
                pred_prob = F.softmax(output, dim=1)[0, pred_class].item()
            
            # Denormalize image for visualization
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_np)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM')
            axes[1].axis('off')
            
            overlayed = GradCAM.overlay_heatmap(img_np, heatmap)
            axes[2].imshow(overlayed)
            axes[2].set_title(f'Pred: {class_names[pred_class]} ({pred_prob*100:.1f}%)\nTrue: {class_names[labels[i]]}')
            axes[2].axis('off')
            
            plt.suptitle(f'Image: {image_ids[i]}')
            plt.tight_layout()
            
            save_path = output_dir / f'gradcam_{image_ids[i]}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            count += 1
            print(f"Generated {count}/{num_samples}", end='\r')
    
    print(f"\n✅ Generated {count} Grad-CAM visualizations in {output_dir}")


if __name__ == '__main__':
    print("Grad-CAM module loaded successfully!")
    print("\nUsage:")
    print("  1. visualize_gradcam() - Single image Grad-CAM")
    print("  2. visualize_multiple_classes() - Show activations for all classes")
    print("  3. batch_gradcam_visualization() - Batch processing")
