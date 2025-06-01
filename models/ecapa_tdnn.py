"""
ECAPA-TDNN model implementation
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Res2Block, AttentiveStatsPooling

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        channels: int = 512,
        embedding_dim: int = 192,
        num_blocks: int = 3,
        scale: int = 8,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        
        self.res2blocks = nn.ModuleList([
            nn.Sequential(
                Res2Block(channels, kernel_size=3, dilation=2**(i % 3), scale=scale),
                nn.BatchNorm1d(channels)
            ) for i in range(num_blocks)
        ])
        
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * num_blocks, channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        
        self.asp = AttentiveStatsPooling(channels)
        
        self.embedding = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        
        res2block_outputs = []
        for block in self.res2blocks:
            x = block(x)
            res2block_outputs.append(x)
        
        x = torch.cat(res2block_outputs, dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.embedding(x)
        
        return x
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward method"""
        return self.forward(x)

class SpeakerClassifier(nn.Module):
    """Speaker classification head with AAM-Softmax"""
    def __init__(
        self, 
        embedding_dim: int = 192, 
        num_speakers: int = 1000,
        margin: float = 0.2,
        scale: float = 30
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        
        self.margin = margin
        self.scale = scale
        self.eps = 1e-6
        
    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)
        cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)
        
        if labels is not None:
            theta = torch.acos(cosine)
            theta_with_margin = torch.zeros_like(theta)
            theta_with_margin.scatter_(1, labels.unsqueeze(1), self.margin)
            cosine = torch.cos(theta + theta_with_margin)
        
        return cosine * self.scale

class SpeakerVerificationModel(nn.Module):
    """Complete speaker verification model"""
    def __init__(
        self,
        input_dim: int = 80,
        channels: int = 512,
        embedding_dim: int = 192,
        num_blocks: int = 3,
        scale: int = 8,
        num_speakers: int = 1000,
        margin: float = 0.2,
        scale_factor: float = 30,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.backbone = ECAPA_TDNN(
            input_dim=input_dim,
            channels=channels,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            scale=scale,
            dropout_rate=dropout_rate
        )
        
        self.classifier = SpeakerClassifier(
            embedding_dim=embedding_dim,
            num_speakers=num_speakers,
            margin=margin,
            scale=scale_factor
        )
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings, labels)
        return logits
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
