"""
Utility functions for the project.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-1 Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Top-1')
    axes[0, 1].plot(history['val_acc'], label='Val Top-1')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Top-1 Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-5 Accuracy
    if 'val_top5' in history and history['val_top5']:
        axes[1, 0].plot(history['val_top5'], label='Val Top-5', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Validation Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Top-10 Accuracy
    if 'val_top10' in history and history['val_top10']:
        axes[1, 1].plot(history['val_top10'], label='Val Top-10', color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Validation Top-10 Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes=None, save_path='confusion_matrix.png'):
    """Plot confusion matrix for a subset of classes."""
    # Use only top 20 classes for readability
    if classes is None:
        class_counts = np.bincount(y_true)
        top_classes = np.argsort(class_counts)[-20:]
        mask = np.isin(y_true, top_classes)
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]
        classes = top_classes
    else:
        y_true_subset = y_true
        y_pred_subset = y_pred
    
    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix (Top 20 Classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def visualize_predictions(model, dataloader, class_names, device, num_samples=5, save_path='predictions.png'):
    """Visualize model predictions on sample videos."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for idx, (frames, labels, video_ids) in enumerate(dataloader):
            if idx >= num_samples:
                break
            
            frames = frames.to(device)
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            
            for i in range(min(len(frames), num_samples - idx)):
                if idx + i >= num_samples:
                    break
                
                true_label = labels[i].item()
                pred_label = outputs[i].argmax().item()
                confidence = probs[i].max().item()
                
                # Get top-5 predictions
                top5_probs, top5_indices = probs[i].topk(5)
                
                ax = axes[idx + i]
                ax.axis('off')
                
                title = f"Video: {video_ids[i]}\n"
                title += f"True: {class_names[true_label] if class_names else true_label}\n"
                title += f"Pred: {class_names[pred_label] if class_names else pred_label} ({confidence:.3f})\n"
                title += "Top-5: " + ", ".join([
                    f"{class_names[idx.item()] if class_names else idx.item()}: {prob:.3f}"
                    for idx, prob in zip(top5_indices, top5_probs)
                ])
                ax.set_title(title, fontsize=10, loc='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


def save_results(results, save_path='results.json'):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, (np.integer, np.floating)):
            results_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {save_path}")


def load_class_names(json_path='WLASL_v0.3.json', num_classes=300):
    """Load class names from WLASL JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gloss_counts = [(entry['gloss'], len(entry['instances'])) for entry in data]
    gloss_counts.sort(key=lambda x: x[1], reverse=True)
    top_k = [g[0] for g in gloss_counts[:num_classes]]
    
    return top_k


def print_model_summary(model):
    """Print model architecture summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size (FP32): {total_params * 4 / (1024**2):.2f} MB")
    
    # Layer-wise breakdown
    print("\nLayer-wise parameter count:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module_params:,} ({module_params/total_params*100:.1f}%)")
    
    print("=" * 60)


if __name__ == '__main__':
    print("Utility functions loaded.")
