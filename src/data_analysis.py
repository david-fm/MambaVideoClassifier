"""
WLASL Dataset Analysis and Exploration
Provides functions to analyze the WLASL dataset structure, class distributions,
and video properties.
"""

import json
import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def load_wlasl_data(json_path='WLASL_v0.3.json'):
    """Load WLASL JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_subset_statistics(data, k=300):
    """Get statistics for top-k glosses."""
    # Sort glosses by number of instances
    gloss_counts = [(entry['gloss'], len(entry['instances'])) for entry in data]
    gloss_counts.sort(key=lambda x: x[1], reverse=True)
    top_k_glosses = [g[0] for g in gloss_counts[:k]]
    
    # Build subset
    subset_stats = {'train': 0, 'val': 0, 'test': 0}
    class_distribution = {}
    signer_ids = set()
    video_lengths = []
    sources = Counter()
    
    for entry in data:
        if entry['gloss'] not in top_k_glosses:
            continue
            
        gloss = entry['gloss']
        class_distribution[gloss] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}
        
        for inst in entry['instances']:
            split = inst['split']
            subset_stats[split] += 1
            class_distribution[gloss][split] += 1
            class_distribution[gloss]['total'] += 1
            signer_ids.add(inst['signer_id'])
            sources[inst['source']] += 1
            
            # Calculate video duration if possible
            frame_start = inst.get('frame_start', 1)
            frame_end = inst.get('frame_end', -1)
            fps = inst.get('fps', 25)
            if frame_end > 0 and frame_start > 0:
                duration = (frame_end - frame_start + 1) / fps
                video_lengths.append(duration)
    
    return {
        'subset_stats': subset_stats,
        'class_distribution': class_distribution,
        'num_signers': len(signer_ids),
        'num_classes': len(top_k_glosses),
        'video_lengths': video_lengths,
        'sources': sources
    }


def plot_class_distribution(class_distribution, save_path='class_distribution.png'):
    """Plot histogram of instances per class."""
    counts = [v['total'] for v in class_distribution.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(counts, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Instances per Class')
    axes[0].set_ylabel('Number of Classes')
    axes[0].set_title('Distribution of Instances per Class')
    axes[0].axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    axes[0].legend()
    
    # Bar plot of top 20 classes
    sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1]['total'], reverse=True)[:20]
    names = [c[0] for c in sorted_classes]
    values = [c[1]['total'] for c in sorted_classes]
    
    axes[1].barh(range(len(names)), values, color='steelblue')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_title('Top 20 Classes by Instance Count')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def plot_split_distribution(subset_stats, save_path='split_distribution.png'):
    """Plot train/val/test split."""
    fig, ax = plt.subplots(figsize=(8, 6))
    splits = list(subset_stats.keys())
    counts = list(subset_stats.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.bar(splits, counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Number of Videos')
    ax.set_title('WLASL300 Dataset Split')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Split distribution plot saved to {save_path}")


def analyze_video_properties(data, video_root='videos', max_samples=100):
    """Analyze actual video properties (resolution, frame count, etc.)."""
    properties = []
    count = 0
    
    for entry in data:
        for inst in entry['instances']:
            if count >= max_samples:
                break
                
            video_id = inst['video_id']
            video_path = os.path.join(video_root, f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                continue
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            properties.append({
                'video_id': video_id,
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'split': inst['split']
            })
            
            cap.release()
            count += 1
            
        if count >= max_samples:
            break
    
    return pd.DataFrame(properties)


def print_summary(stats):
    """Print formatted summary of dataset statistics."""
    print("=" * 60)
    print("WLASL DATASET SUMMARY")
    print("=" * 60)
    print(f"Subset: WLASL{stats['num_classes']}")
    print(f"Total classes: {stats['num_classes']}")
    print(f"Total videos: {sum(stats['subset_stats'].values())}")
    print(f"  - Train: {stats['subset_stats']['train']}")
    print(f"  - Val: {stats['subset_stats']['val']}")
    print(f"  - Test: {stats['subset_stats']['test']}")
    print(f"Number of signers: {stats['num_signers']}")
    
    counts = [v['total'] for v in stats['class_distribution'].values()]
    print(f"\nInstances per class:")
    print(f"  - Min: {min(counts)}")
    print(f"  - Max: {max(counts)}")
    print(f"  - Mean: {np.mean(counts):.1f}")
    print(f"  - Median: {np.median(counts):.1f}")
    
    if stats['video_lengths']:
        print(f"\nVideo durations (from annotations):")
        print(f"  - Mean: {np.mean(stats['video_lengths']):.2f}s")
        print(f"  - Median: {np.median(stats['video_lengths']):.2f}s")
        print(f"  - Min: {min(stats['video_lengths']):.2f}s")
        print(f"  - Max: {max(stats['video_lengths']):.2f}s")
    
    print(f"\nTop 10 sources:")
    for source, count in stats['sources'].most_common(10):
        print(f"  - {source}: {count}")
    print("=" * 60)


if __name__ == '__main__':
    # Example usage
    data = load_wlasl_data('WLASL_v0.3.json')
    stats = get_subset_statistics(data, k=300)
    print_summary(stats)
    plot_class_distribution(stats['class_distribution'])
    plot_split_distribution(stats['subset_stats'])
