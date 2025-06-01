"""
Feature extraction modules
"""
import torch
import torch.nn as nn
import torchaudio
from .modules import PreEmphasis

class FeatureExtractor(nn.Module):
    """Extract log mel filterbank features from audio"""
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512, 
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        apply_preemphasis: bool = True,
        preemphasis_coef: float = 0.97
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=0,
            f_max=sample_rate/2,
            n_mels=n_mels,
            power=2.0,
        )
        
        self.apply_preemphasis = apply_preemphasis
        if apply_preemphasis:
            self.preemphasis = PreEmphasis(preemphasis_coef)
        
        self.log_eps = 1e-6
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.apply_preemphasis:
            waveform = self.preemphasis(waveform)
        
        mel_spec = self.mel_spec(waveform)
        log_mel_spec = torch.log(mel_spec + self.log_eps)
        
        return log_mel_spec
