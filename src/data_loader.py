"""
WLASL Data Loader
Handles loading, preprocessing, and batching of WLASL video data.
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WLASLDataset(Dataset):
    """
    WLASL Dataset for isolated sign language recognition.
    
    Args:
        json_path: Path to WLASL_v0.3.json
        video_root: Directory containing video files
        split: 'train', 'val', or 'test'
        num_classes: Number of classes to use (100, 300, 1000, or 2000)
        num_frames: Number of frames to sample per video
        frame_size: Target frame size (height, width)
        transform: Optional torchvision transforms
    """
    
    def __init__(self, json_path, video_root, split='train', num_classes=300,
                 num_frames=25, frame_size=(224, 224), transform=None,
                 video_root_backup=None):
        self.video_root = video_root
        self.video_root_backup = video_root_backup
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        
        # Load annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Select top-k classes
        gloss_counts = [(entry['gloss'], len(entry['instances'])) for entry in data]
        gloss_counts.sort(key=lambda x: x[1], reverse=True)
        self.top_k_glosses = [g[0] for g in gloss_counts[:num_classes]]
        
        # Create class to index mapping
        self.class_to_idx = {gloss: idx for idx, gloss in enumerate(self.top_k_glosses)}
        
        # Build dataset samples
        self.samples = []
        for entry in data:
            if entry['gloss'] not in self.top_k_glosses:
                continue
            
            gloss = entry['gloss']
            label = self.class_to_idx[gloss]
            
            for inst in entry['instances']:
                # Map split
                inst_split = inst['split']
                if split == 'train' and inst_split not in ['train', 'val']:
                    continue
                elif split in ['val', 'test'] and inst_split != split:
                    continue
                
                video_id = inst['video_id']
                video_path = os.path.join(video_root, f"{video_id}.mp4")
                
                # Check backup if primary doesn't exist
                if not os.path.exists(video_path):
                    if self.video_root_backup is not None:
                        video_path = os.path.join(self.video_root_backup, f"{video_id}.mp4")
                    if not os.path.exists(video_path):
                        continue
                
                self.samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'label': label,
                    'gloss': gloss,
                    'bbox': inst.get('bbox', None),
                    'frame_start': inst.get('frame_start', 1),
                    'frame_end': inst.get('frame_end', -1),
                    'fps': inst.get('fps', 25)
                })
        
        print(f"WLASL{num_classes} {split}: {len(self.samples)} samples loaded")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        
        # Load and preprocess video
        frames = self.load_video(video_path)
        
        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label, sample['video_id']
    
    def load_video(self, video_path):
        """Load video and sample frames uniformly."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # Return dummy frames if video can't be opened
            return torch.zeros(3, self.num_frames, *self.frame_size)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return torch.zeros(3, self.num_frames, *self.frame_size)
        
        # Sample frame indices uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read fails, use last valid frame or zeros
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, self.frame_size)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        # Stack frames: (T, H, W, C) -> (C, T, H, W)
        frames = np.stack(frames, axis=0)  # (T, H, W, C)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        
        return torch.from_numpy(frames).float()


class VideoTransform:
    """Transform pipeline for video data."""
    
    def __init__(self, mode='train', frame_size=224):
        self.mode = mode
        self.frame_size = frame_size
        
        if mode == 'train':
            # Training transforms
            self.transforms = transforms.Compose([
                RandomHorizontalFlipVideo(p=0.5),
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/test transforms
            self.transforms = transforms.Compose([
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, frames):
        return self.transforms(frames)


class RandomHorizontalFlipVideo:
    """Randomly flip video horizontally."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, frames):
        # frames: (C, T, H, W)
        if random.random() < self.p:
            frames = torch.flip(frames, dims=[-1])
        return frames


class NormalizeVideo:
    """Normalize video frames with mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, frames):
        # frames: (C, T, H, W)
        self.mean = self.mean.to(frames.device)
        self.std = self.std.to(frames.device)
        return (frames - self.mean) / self.std


def get_data_loaders(json_path, video_root, num_classes=300, num_frames=25,
                     frame_size=224, batch_size=32, num_workers=4,
                     video_root_backup=None):
    """
    Create train, validation, and test data loaders.
    
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    train_dataset = WLASLDataset(
        json_path=json_path,
        video_root=video_root,
        split='train',
        num_classes=num_classes,
        num_frames=num_frames,
        frame_size=(frame_size, frame_size),
        transform=VideoTransform(mode='train', frame_size=frame_size),
        video_root_backup=video_root_backup
    )
    
    val_dataset = WLASLDataset(
        json_path=json_path,
        video_root=video_root,
        split='val',
        num_classes=num_classes,
        num_frames=num_frames,
        frame_size=(frame_size, frame_size),
        transform=VideoTransform(mode='val', frame_size=frame_size),
        video_root_backup=video_root_backup
    )
    
    test_dataset = WLASLDataset(
        json_path=json_path,
        video_root=video_root,
        split='test',
        num_classes=num_classes,
        num_frames=num_frames,
        frame_size=(frame_size, frame_size),
        transform=VideoTransform(mode='val', frame_size=frame_size),
        video_root_backup=video_root_backup
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Quick test
    loaders = get_data_loaders(
        json_path='WLASL_v0.3.json',
        video_root='videos',
        num_classes=300,
        num_frames=25,
        batch_size=4
    )
    
    for frames, labels, video_ids in loaders['train']:
        print(f"Batch shape: {frames.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Video IDs: {video_ids}")
        break
