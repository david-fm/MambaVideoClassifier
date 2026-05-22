"""
CNN-Mamba Model for WLASL Sign Language Recognition
Implements MobileNetV3-Large + Selective State Space Model (Mamba) architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


class SelectiveSSM(nn.Module):
    """
    Simplified Selective State Space Model (Mamba) implementation in pure PyTorch.
    
    This implements the core selective SSM mechanism:
    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C * h_t
    
    Where A_bar, B_bar are discretized versions with input-dependent step sizes.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize A as S4-style (structured state space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=seq_len, D=d_model
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Apply convolution for local mixing
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L]  # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        y = self.ssm(x_conv)  # (B, L, d_inner)
        
        # Gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)  # (B, L, D)
        
        return output
    
    def ssm(self, x):
        """
        Run the selective SSM.
        
        Args:
            x: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, _ = x.shape
        
        # Get discretization parameters
        x_dbl = self.x_proj(x)  # (B, L, d_state*2 + 1)
        delta, B, C = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        
        # Discretize step size
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Get A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize: A_bar = exp(delta * A)
        # For efficiency, we use a sequential scan
        # In practice, parallel scan would be used, but sequential is fine for small L
        y = self.selective_scan(x, delta, A, B, C)
        
        return y
    
    def selective_scan(self, x, delta, A, B, C):
        """
        Sequential selective scan implementation.
        
        Args:
            x: (B, L, d_inner)
            delta: (B, L, d_inner)
            A: (d_inner, d_state)
            B: (B, L, d_state)
            C: (B, L, d_state)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, d_in = x.shape
        d_state = A.size(1)
        
        # Initialize state
        h = torch.zeros(B, d_in, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            # Discretize
            delta_t = delta[:, t, :]  # (B, d_in)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_in, d_state)
            B_bar = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            
            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            
            # Output: y = C * h
            y_t = torch.sum(C[:, t, :].unsqueeze(1) * h, dim=-1)  # (B, d_in)
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_in)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection and layer normalization.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return x + residual


class MobileNetV3_Mamba(nn.Module):
    """
    CNN-Mamba hybrid architecture for video sign language recognition.
    
    Architecture:
    1. MobileNetV3-Large as spatial encoder (per-frame feature extraction)
    2. Mamba blocks as temporal decoder (sequence modeling)
    3. Classification head with temporal pooling
    """
    
    def __init__(self, num_classes=300, num_frames=25, d_model=256, 
                 n_mamba_layers=2, d_state=16, dropout=0.5, 
                 pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.d_model = d_model
        
        # Spatial encoder: MobileNetV3-Large
        mobilenet = mobilenet_v3_large(pretrained=pretrained)
        # Remove classifier and last pooling
        self.spatial_encoder = nn.Sequential(*list(mobilenet.features.children()))
        
        # Global pooling after features to get per-frame embedding
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # Get feature dimension (MobileNetV3-Large outputs 960 channels)
        self.spatial_dim = 960
        
        # Projection to d_model
        self.input_proj = nn.Linear(self.spatial_dim, d_model)
        
        # Temporal decoder: Mamba blocks
        self.temporal_decoder = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-pretrained layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_frame_features(self, frames):
        """
        Extract spatial features from frames.
        
        Args:
            frames: (B, C, T, H, W)
        Returns:
            features: (B, T, D_spatial)
        """
        B, C, T, H, W = frames.shape
        
        # Process each frame independently: (B*T, C, H, W)
        frames_flat = frames.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        # Extract features
        features = self.spatial_encoder(frames_flat)  # (B*T, 960, H', W')
        features = self.spatial_pool(features)  # (B*T, 960, 1, 1)
        features = features.view(B, T, self.spatial_dim)  # (B, T, 960)
        
        return features
    
    def forward(self, frames):
        """
        Forward pass.
        
        Args:
            frames: (B, C, T, H, W) - batch of videos
        Returns:
            logits: (B, num_classes)
        """
        # Spatial encoding
        x = self.extract_frame_features(frames)  # (B, T, 960)
        
        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Temporal decoding
        for block in self.temporal_decoder:
            x = block(x)  # (B, T, d_model)
        
        # Temporal pooling: (B, T, d_model) -> (B, d_model, T) -> (B, d_model, 1) -> (B, d_model)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.temporal_pool(x).squeeze(-1)  # (B, d_model)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits
    
    def get_flops(self, input_shape=(1, 3, 32, 224, 224)):
        """Estimate FLOPs using fvcore if available."""
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randn(input_shape)
            flops = FlopCountAnalysis(self, dummy_input)
            return flops.total()
        except ImportError:
            print("fvcore not installed. Install with: pip install fvcore")
            return None
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=300, num_frames=25, d_model=256, 
                 n_mamba_layers=2, pretrained=True):
    """
    Factory function to create the model.
    
    Args:
        num_classes: Number of output classes
        num_frames: Number of frames per video
        d_model: Hidden dimension of Mamba layers
        n_mamba_layers: Number of Mamba blocks
        pretrained: Use ImageNet pretrained weights for MobileNetV3
    
    Returns:
        model: MobileNetV3_Mamba instance
    """
    model = MobileNetV3_Mamba(
        num_classes=num_classes,
        num_frames=num_frames,
        d_model=d_model,
        n_mamba_layers=n_mamba_layers,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # Quick test
    model = create_model(num_classes=300, num_frames=25, d_model=256, n_mamba_layers=2)
    
    # Count parameters
    n_params = model.count_parameters()
    print(f"Total parameters: {n_params:,}")
    print(f"Model size (FP32): {n_params * 4 / (1024**2):.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 300)")
