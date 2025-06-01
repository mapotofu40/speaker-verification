"""
Utilities for evaluation and metrics
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve
import torch.nn.functional as F
from typing import Tuple

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER)
    
    Args:
        scores: Similarity scores between pairs
        labels: Binary labels (1 for same speaker, 0 for different speaker)
    
    Returns:
        Tuple of (EER value, EER threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the index where the EER occurs
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    
    # EER value and threshold at this point
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold

def cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between embeddings
    """
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
    return torch.sum(embeddings1_norm * embeddings2_norm, dim=-1)

def verify_speakers(embedding1: torch.Tensor, embedding2: torch.Tensor, threshold: float = 0.5) -> bool:
    """
    Verify if two embeddings belong to the same speaker
    """
    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item() > threshold
