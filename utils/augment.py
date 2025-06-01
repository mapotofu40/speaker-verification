"""
Audio augmentation utilities for speaker verification
"""
import torch
import torchaudio
import random
from typing import Tuple, Optional

class AudioAugmentor:
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb: bool = True,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5, 20),
        reverb_prob: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.speed_perturb = speed_perturb
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.reverb_prob = reverb_prob
        
        # Initialize RIR (Room Impulse Response) convolution
        self._load_default_rir()
    
    def _load_default_rir(self):
        """Load a default RIR or generate a simple one"""
        # Simple exponential decay as default RIR
        t = torch.linspace(0, 0.5, 8000)
        self.default_rir = torch.exp(-4 * t).unsqueeze(0)
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR"""
        if random.random() > self.noise_prob:
            return waveform
            
        # Calculate signal power
        signal_power = waveform.norm(p=2)
        
        # Calculate noise power based on SNR
        snr = 10 ** (snr_db / 10)
        noise_power = signal_power / snr
        
        # Generate noise
        noise = torch.randn_like(waveform) * (noise_power / waveform.size(-1)**0.5)
        
        return waveform + noise
    
    def apply_speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation"""
        if not self.speed_perturb or random.random() > 0.5:
            return waveform
            
        speed_factor = random.uniform(0.9, 1.1)
        
        # Resample to change speed
        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            int(self.sample_rate * speed_factor)
        )
        
        # Apply speed change and resample back
        waveform = resampler(waveform)
        resampler = torchaudio.transforms.Resample(
            int(self.sample_rate * speed_factor),
            self.sample_rate
        )
        return resampler(waveform)
    
    def apply_reverb(self, waveform: torch.Tensor, rir: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply reverberation using room impulse response"""
        if random.random() > self.reverb_prob:
            return waveform
            
        rir = rir if rir is not None else self.default_rir
        
        # Normalize RIR
        rir = rir / torch.norm(rir, p=2)
        
        # Apply convolution
        reverb_audio = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            rir.unsqueeze(0),
            padding=rir.shape[-1]-1
        )
        
        return reverb_audio.squeeze(0)
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations randomly"""
        # Speed perturbation
        waveform = self.apply_speed_perturbation(waveform)
        
        # Add noise
        if random.random() < self.noise_prob:
            snr = random.uniform(*self.noise_snr_range)
            waveform = self.add_noise(waveform, snr)
        
        # Apply reverb
        waveform = self.apply_reverb(waveform)
        
        return waveform
