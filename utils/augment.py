"""
Audio augmentation utilities for speaker verification
"""
import torch
import torchaudio
import random
import os
from typing import Tuple, Optional, List, Dict
from pathlib import Path

class MUSANNoiseDataset:
    """Dataset class for loading and managing MUSAN noise samples"""
    def __init__(self, musan_path: str, sample_rate: int = 16000):
        self.musan_path = Path(musan_path)
        self.sample_rate = sample_rate
        self.noise_files: Dict[str, List[str]] = {
            'noise': [],
            'music': [],
            'speech': []
        }
        self._load_noise_files()
    
    def _load_noise_files(self):
        """Load paths to all noise files from MUSAN dataset"""
        for category in self.noise_files.keys():
            category_path = self.musan_path / category
            if category_path.exists():
                self.noise_files[category].extend([
                    str(f) for f in category_path.rglob("*.wav")
                ])
    
    def get_random_noise(self, category: str, duration: float) -> torch.Tensor:
        """Get a random noise sample of specified duration from given category
        
        Args:
            category: One of 'noise', 'music', or 'speech'
            duration: Desired duration in seconds
            
        Returns:
            Tensor containing noise sample resampled to match sample_rate
        """
        if not self.noise_files[category]:
            raise ValueError(f"No noise files found for category {category}")
            
        # Select random file
        noise_path = random.choice(self.noise_files[category])
        
        # Load and resample if needed
        waveform, sr = torchaudio.load(noise_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # Convert duration to samples
        target_length = int(duration * self.sample_rate)
        
        # If noise is shorter than target, repeat it
        if waveform.size(1) < target_length:
            num_repeats = (target_length + waveform.size(1) - 1) // waveform.size(1)
            waveform = waveform.repeat(1, num_repeats)
        
        # Randomly crop to desired length
        start = random.randint(0, waveform.size(1) - target_length)
        waveform = waveform[:, start:start + target_length]
        
        return waveform

class AudioAugmentor:
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb: bool = True,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5, 20),
        reverb_prob: float = 0.5,
        musan_path: Optional[str] = None,
        musan_noise_prob: float = 0.3,
        noise_types: Optional[Dict[str, float]] = None
    ):
        self.sample_rate = sample_rate
        self.speed_perturb = speed_perturb
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.reverb_prob = reverb_prob
        self.musan_noise_prob = musan_noise_prob
        self.noise_types = noise_types or {
            'noise': 0.4,
            'music': 0.3,
            'speech': 0.3
        }
        
        # Initialize MUSAN dataset if path provided
        self.musan = None
        if musan_path:
            self.musan = MUSANNoiseDataset(musan_path, sample_rate)
        
        # Initialize RIR (Room Impulse Response) convolution
        self._load_default_rir()
    
    def _load_default_rir(self):
        """Load a default RIR or generate a simple one"""
        # Simple exponential decay as default RIR
        t = torch.linspace(0, 0.5, 8000)
        self.default_rir = torch.exp(-4 * t).unsqueeze(0)
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float, noise_type: Optional[str] = None) -> torch.Tensor:
        """Add noise at specified SNR
        
        Args:
            waveform: Input audio waveform
            snr_db: Signal-to-noise ratio in dB
            noise_type: Type of noise to add ('noise', 'music', 'speech', or None for Gaussian)
            
        Returns:
            Augmented waveform with added noise
        """
        if random.random() > self.noise_prob:
            return waveform
            
        # Calculate signal power
        signal_power = waveform.norm(p=2)
        
        # Calculate desired noise power based on SNR
        snr = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr
        
        if self.musan and noise_type:
            # Get noise from MUSAN dataset
            duration = waveform.size(-1) / self.sample_rate
            noise = self.musan.get_random_noise(noise_type, duration)
            
            # Normalize noise power
            current_noise_power = noise.norm(p=2)
            noise_scalar = (target_noise_power / current_noise_power) ** 0.5
            noise = noise * noise_scalar
        else:
            # Generate Gaussian noise
            noise = torch.randn_like(waveform) * (target_noise_power / waveform.size(-1)**0.5)
        
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
            
            # If MUSAN dataset is available, use it with configured probability
            if self.musan and random.random() < self.musan_noise_prob:
                # Choose noise type based on configured probabilities
                noise_probs = list(self.noise_types.values())
                noise_types = list(self.noise_types.keys())
                noise_type = random.choices(noise_types, weights=noise_probs, k=1)[0]
                waveform = self.add_noise(waveform, snr, noise_type)
            else:
                # Fallback to Gaussian noise
                waveform = self.add_noise(waveform, snr)
        
        # Apply reverb
        waveform = self.apply_reverb(waveform)
        
        return waveform
