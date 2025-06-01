"""
Script for speaker verification inference
"""
import os
import torch
import logging
import argparse
from pathlib import Path
import torchaudio

from models import SpeakerVerificationModel, FeatureExtractor
from utils import verify_speakers
from config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_embedding(
    model: SpeakerVerificationModel,
    feature_extractor: FeatureExtractor,
    audio_path: str,
    device: torch.device
) -> torch.Tensor:
    """Extract speaker embedding from an audio file"""
    try:
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        
        if sr != DEFAULT_CONFIG['sample_rate']:
            waveform = torchaudio.transforms.Resample(sr, DEFAULT_CONFIG['sample_rate'])(waveform)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(waveform.to(device))
            embedding = model.extract_embedding(features)
        
        return embedding
        
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {e}")
        raise

def verify_pair(
    model_path: str,
    audio1_path: str,
    audio2_path: str,
    threshold: float = 0.5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> bool:
    """
    Verify if two audio files contain the same speaker
    
    Args:
        model_path: Path to the trained model checkpoint
        audio1_path: Path to first audio file
        audio2_path: Path to second audio file
        threshold: Similarity threshold for verification
        device: Device to run inference on
        
    Returns:
        Boolean indicating if the speakers are the same
    """
    try:
        device = torch.device(device)
        
        # Load model
        model = SpeakerVerificationModel(
            input_dim=DEFAULT_CONFIG['n_mels'],
            channels=DEFAULT_CONFIG['channels'],
            embedding_dim=DEFAULT_CONFIG['embedding_dim']
        )
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        # Create feature extractor
        feature_extractor = FeatureExtractor(
            sample_rate=DEFAULT_CONFIG['sample_rate'],
            n_mels=DEFAULT_CONFIG['n_mels']
        )
        
        # Extract embeddings
        embedding1 = extract_embedding(model, feature_extractor, audio1_path, device)
        embedding2 = extract_embedding(model, feature_extractor, audio2_path, device)
        
        # Verify speakers
        result = verify_speakers(embedding1, embedding2, threshold)
        
        return result
        
    except Exception as e:
        logging.error(f"Verification failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Verify if two audio files contain the same speaker")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--audio1", type=str, required=True, help="Path to first audio file")
    parser.add_argument("--audio2", type=str, required=True, help="Path to second audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help="Device to run inference on")
    
    args = parser.parse_args()
    
    result = verify_pair(
        model_path=args.model_path,
        audio1_path=args.audio1,
        audio2_path=args.audio2,
        threshold=args.threshold,
        device=args.device
    )
    
    if result:
        print("Same speaker")
    else:
        print("Different speakers")

if __name__ == "__main__":
    main()
