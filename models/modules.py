"""
Base modules for the ECAPA-TDNN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreEmphasis(nn.Module):
    """Pre-emphasis filter to emphasize higher frequencies"""
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, 'Input must be 2D tensor of shape (batch, time)'
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x.unsqueeze(1), self.flipped_filter).squeeze(1)

class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention"""
    def __init__(self, channels: int, bottleneck_ratio: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // bottleneck_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // bottleneck_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)

class Res2Block(nn.Module):
    """Res2Net block with Squeeze-and-Excitation"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, 
                 scale: int = 8, se_ratio: int = 16):
        super().__init__()
        self.scale = scale
        assert channels % scale == 0, "channels must be divisible by scale"
        self.width = channels // scale
        self.convs = nn.ModuleList()
        
        padding = (kernel_size - 1) * dilation // 2
        for _ in range(scale):
            self.convs.append(
                nn.Conv1d(self.width, self.width, kernel_size=kernel_size, 
                         dilation=dilation, padding=padding)
            )
        
        self.se = SEModule(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        spx = torch.split(x, self.width, dim=1)
        out = [spx[0]]
        
        for i in range(1, self.scale):
            sp = spx[i] if i == 1 else spx[i] + out[-1]
            sp = self.convs[i](sp)
            out.append(sp)
        
        out = torch.cat(out, dim=1)
        out = self.se(out)
        return out + residual

class AttentiveStatsPooling(nn.Module):
    """Attentive Statistics Pooling layer"""
    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        mean = torch.sum(x * attention_weights, dim=2)
        var = torch.sum(attention_weights * (x - mean.unsqueeze(2)) ** 2, dim=2)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=1)
