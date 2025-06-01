"""
Default configuration for the speaker verification system
"""
from pathlib import Path

# Base configuration
BASE_CONFIG = {
    # Audio processing
    'audio': {
        'sample_rate': 16000,
        'n_mels': 80,
        'n_fft': 512,
        'win_length': 400,
        'hop_length': 160,
        'max_frames': 300,
        'apply_preemphasis': True,
        'preemphasis_coef': 0.97,
    },
    
    # Model architecture
    'model': {
        'channels': 512,
        'embedding_dim': 192,
        'num_blocks': 3,
        'scale': 8,
        'dropout_rate': 0.2,
        'se_ratio': 16,
    },
    
    # Training settings
    'training': {
        'batch_size': 32,
        'num_epochs': 150,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 10,
        },
        'gradient_clip': 3.0,
        'early_stopping_patience': 10,
    },
    
    # Loss function
    'loss': {
        'margin': 0.2,
        'scale_factor': 30,
    },
    
    # Data augmentation
    'augment': {
        'enabled': True,
        'speed_perturb': True,
        'noise_prob': 0.5,
        'noise_snr_range': [5, 20],
        'reverb_prob': 0.5,
    },
    
    # Inference
    'inference': {
        'threshold': 0.5,
        'batch_size': 32,
    },
    
    # Paths and caching
    'paths': {
        'checkpoint_dir': str(Path('./checkpoints')),
        'log_dir': str(Path('./logs')),
        'cache_dir': str(Path('./cache')),
    },
    
    # Feature caching
    'cache': {
        'enabled': True,
        'max_size_gb': 50,
    },
    
    # Logging
    'logging': {
        'level': 'INFO',
        'use_tensorboard': True,
    },
}

# Environment-specific configurations
COLAB_CONFIG = BASE_CONFIG.copy()
COLAB_CONFIG.update({
    'paths': {
        'checkpoint_dir': '/content/drive/MyDrive/speaker_verification/checkpoints',
        'log_dir': '/content/drive/MyDrive/speaker_verification/logs',
        'cache_dir': '/content/cache',
    }
})
