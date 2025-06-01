"""
Evaluation script for the speaker verification model
"""
import os
import torch
import logging
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SpeakerVerificationModel, FeatureExtractor
from utils import TestDataset, collate_fn, compute_eer, cosine_similarity
from config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(
    model_path: str,
    test_pairs_file: str,
    wav_dir: str,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Evaluate the speaker verification model on test pairs
    
    Args:
        model_path: Path to the trained model checkpoint
        test_pairs_file: File containing test pairs
        wav_dir: Directory containing wav files
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Equal Error Rate (EER)
    """
    try:
        # Load model
        model = SpeakerVerificationModel(
            input_dim=DEFAULT_CONFIG['n_mels'],
            channels=DEFAULT_CONFIG['channels'],
            embedding_dim=DEFAULT_CONFIG['embedding_dim']
        )
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        # Create dataset and loader
        feature_extractor = FeatureExtractor(
            sample_rate=DEFAULT_CONFIG['sample_rate'],
            n_mels=DEFAULT_CONFIG['n_mels']
        )
        
        test_dataset = TestDataset(
            test_pairs_file,
            wav_dir,
            feature_extractor=feature_extractor
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Evaluate
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                features1 = batch['features1'].to(device)
                features2 = batch['features2'].to(device)
                labels = batch['labels'].to(device)
                
                # Extract embeddings
                embeddings1 = model.extract_embedding(features1)
                embeddings2 = model.extract_embedding(features2)
                
                # Compute similarity scores
                scores = cosine_similarity(embeddings1, embeddings2)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute EER
        eer, threshold = compute_eer(all_scores, all_labels)
        logging.info(f"Equal Error Rate: {eer*100:.2f}% at threshold {threshold:.3f}")
        
        return eer
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_pairs", type=str, required=True, help="Path to test pairs file")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing wav files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        test_pairs_file=args.test_pairs,
        wav_dir=args.wav_dir,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
