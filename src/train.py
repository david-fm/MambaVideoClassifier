"""
Training and Evaluation Utilities
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """
    Trainer class for CNN-Mamba model.
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, num_classes=300):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': [],
            'val_top10': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        for batch_idx, (frames, labels, _) in enumerate(self.train_loader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            acc1, acc5 = self.accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), frames.size(0))
            top1.update(acc1.item(), frames.size(0))
            top5.update(acc5.item(), frames.size(0))
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {losses.avg:.4f} "
                      f"Top-1: {top1.avg:.2f}% "
                      f"Top-5: {top5.avg:.2f}%")
        
        return losses.avg, top1.avg, top5.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        top10 = AverageMeter()
        
        all_preds = []
        all_labels = []
        
        for frames, labels, _ in self.val_loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Metrics
            acc1, acc5, acc10 = self.accuracy(outputs, labels, topk=(1, 5, 10))
            losses.update(loss.item(), frames.size(0))
            top1.update(acc1.item(), frames.size(0))
            top5.update(acc5.item(), frames.size(0))
            top10.update(acc10.item(), frames.size(0))
            
            # Store for per-class accuracy
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Per-class accuracy
        per_class_acc = self.per_class_accuracy(all_preds, all_labels)
        
        print(f"\nValidation Epoch {epoch}:")
        print(f"  Loss: {losses.avg:.4f}")
        print(f"  Top-1: {top1.avg:.2f}%")
        print(f"  Top-5: {top5.avg:.2f}%")
        print(f"  Top-10: {top10.avg:.2f}%")
        print(f"  Per-class Acc: {per_class_acc:.2f}%")
        
        return losses.avg, top1.avg, top5.avg, top10.avg, per_class_acc
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        
        top1 = AverageMeter()
        top5 = AverageMeter()
        top10 = AverageMeter()
        
        all_preds = []
        all_labels = []
        inference_times = []
        
        for frames, labels, _ in test_loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Measure inference time
            start = time.time()
            outputs = self.model(frames)
            if torch.cuda.is_available():
                if isinstance(self.model, nn.DataParallel):
                    torch.cuda.synchronize(self.model.output_device)
                else:
                    torch.cuda.synchronize()
            end = time.time()
            
            batch_time = (end - start) / frames.size(0)
            inference_times.append(batch_time)
            
            # Metrics
            acc1, acc5, acc10 = self.accuracy(outputs, labels, topk=(1, 5, 10))
            top1.update(acc1.item(), frames.size(0))
            top5.update(acc5.item(), frames.size(0))
            top10.update(acc10.item(), frames.size(0))
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        per_class_acc = self.per_class_accuracy(all_preds, all_labels)
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        print("\n" + "=" * 60)
        print("TEST SET RESULTS")
        print("=" * 60)
        print(f"Top-1 Accuracy: {top1.avg:.2f}%")
        print(f"Top-5 Accuracy: {top5.avg:.2f}%")
        print(f"Top-10 Accuracy: {top10.avg:.2f}%")
        print(f"Per-class Accuracy: {per_class_acc:.2f}%")
        print(f"Avg Inference Time: {avg_inference_time:.2f} ms/video")
        print(f"Throughput: {1000/avg_inference_time:.1f} videos/sec")
        print("=" * 60)
        
        return {
            'top1': top1.avg,
            'top5': top5.avg,
            'top10': top10.avg,
            'per_class_acc': per_class_acc,
            'inference_time_ms': avg_inference_time
        }
    
    @torch.no_grad()
    def evaluate_cpu_speed(self, test_loader, num_runs=100):
        """Evaluate CPU inference speed on a single video."""
        self.model.cpu()
        self.model.eval()
        
        # Get a single video from test set
        frames, labels, _ = next(iter(test_loader))
        video = frames[0:1].cpu()  # batch_size=1
        
        # Warm-up
        for _ in range(10):
            _ = self.model(video)
        
        # Timing
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.model(video)
            end = time.time()
            times.append((end - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nCPU Inference Speed (single video):")
        print(f"  Runs: {num_runs}")
        print(f"  Avg: {avg_time:.2f} ms")
        print(f"  Std: {std_time:.2f} ms")
        print(f"  FPS: {1000/avg_time:.1f}")
        
        return avg_time, std_time
    
    def accuracy(self, output, target, topk=(1,)):
        """Compute top-k accuracy."""
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def per_class_accuracy(self, preds, labels):
        """Compute mean per-class accuracy."""
        preds = np.array(preds)
        labels = np.array(labels)
        
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred, label in zip(preds, labels):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        
        accs = []
        for c in class_total:
            accs.append(class_correct[c] / class_total[c])
        
        return np.mean(accs) * 100 if accs else 0.0
    
    def save_checkpoint(self, epoch, filename='checkpoint.pth'):
        """Save model checkpoint."""
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel state dict loading
        is_dp = isinstance(self.model, nn.DataParallel)
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        
        if is_dp and not has_module_prefix:
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_dp and has_module_prefix:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filename}")
        return checkpoint['epoch']


def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    Create optimizer with different learning rates for pretrained and new layers.
    """
    # Separate parameters
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'spatial_encoder' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': lr}
    ], weight_decay=weight_decay)
    
    return optimizer


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print("Training utilities loaded successfully.")
