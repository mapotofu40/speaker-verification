# Common configuration settings
DEFAULT_CONFIG = {
    # Audio settings
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 512,
    'win_length': 400,
    'hop_length': 160,
    'max_frames': 300,
    
    # Model architecture
    'channels': 512,
    'embedding_dim': 192,
    'num_blocks': 3,
    'scale': 8,
    'margin': 0.2,
    'scale_factor': 30,
    'dropout_rate': 0.2,
    
    # Training settings
    'batch_size': 32,
    'num_epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'checkpoint_interval': 5,
    
    # Feature extraction
    'apply_preemphasis': True,
    'preemphasis_coef': 0.97,
    
    # Paths and caching
    'use_cache': True,
    'embedding_cache': "embedding_cache.pt"
}
