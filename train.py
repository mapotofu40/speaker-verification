"""
Main script for training and evaluation
"""
import os
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from config import DEFAULT_CONFIG
from models.feature_extractor import FeatureExtractor
from models.ecapa_tdnn import SpeakerVerificationModel
from utils.data import VietnamCelebDataset, collate_fn
from utils.training import train_model
from utils.metrics import compute_eer

def main():
    parser = ArgumentParser(description="Train and evaluate ECAPA-TDNN for speaker verification")
    
    # Data paths
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to speaker metadata file")
    parser.add_argument("--utterance_file", type=str, required=True, help="Path to utterance list file")
    
    # Model configuration
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_CONFIG['embedding_dim'])
    parser.add_argument("--channels", type=int, default=DEFAULT_CONFIG['channels'])
    parser.add_argument("--num_blocks", type=int, default=DEFAULT_CONFIG['num_blocks'])
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG['num_epochs'])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")
    
    # Feature extraction
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_CONFIG['sample_rate'])
    parser.add_argument("--n_mels", type=int, default=DEFAULT_CONFIG['n_mels'])
    parser.add_argument("--use_cache", action="store_true", help="Enable feature caching")
    parser.add_argument("--cache_dir", type=str, help="Directory to store cached features")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels
    )
    
    # Create dataset
    dataset = VietnamCelebDataset(
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        utterance_file=args.utterance_file,
        feature_extractor=feature_extractor,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = SpeakerVerificationModel(
        input_dim=args.n_mels,
        channels=args.channels,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_speakers=len(dataset.speaker_to_idx)
    )
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
