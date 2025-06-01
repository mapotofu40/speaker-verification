"""
Data loading and processing utilities
"""
import os
import os.path as osp
import torch
import torchaudio
import hashlib
import pandas as pd
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from models.feature_extractor import FeatureExtractor

class VietnamCelebDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        utterance_file: str,
        max_frames: int = 300,
        sample_rate: int = 16000,
        feature_extractor: Optional[FeatureExtractor] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False
    ):
        super().__init__()
        
        self.data_root = data_root
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        
        # Set up cache
        self.cache_dir = cache_dir if cache_dir else osp.join(data_root, "feature_cache")
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load metadata and utterances
        self.metadata = pd.read_csv(metadata_file, sep='\t')
        self.utterances = []
        with open(utterance_file, 'r') as f:
            for line in f:
                speaker_id, wav_file = line.strip().split('\t')
                self.utterances.append((speaker_id, wav_file))
        
        # Create speaker mapping
        self.speaker_to_idx = {
            speaker_id: idx for idx, speaker_id in enumerate(self.metadata['speaker_id'].unique())
        }
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor if feature_extractor else FeatureExtractor()
    
    def _get_cache_path(self, speaker_id: str, wav_file: str) -> str:
        uid = f"{speaker_id}_{wav_file}"
        hash_str = hashlib.md5(uid.encode()).hexdigest()
        return osp.join(self.cache_dir, f"{hash_str}.pt")
    
    def __len__(self) -> int:
        return len(self.utterances)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        speaker_id, wav_file = self.utterances[idx]
        speaker_idx = self.speaker_to_idx[speaker_id]
        
        # Try cache first
        if self.use_cache:
            cache_path = self._get_cache_path(speaker_id, wav_file)
            if osp.exists(cache_path):
                try:
                    features = torch.load(cache_path)
                    return {
                        'features': features,
                        'speaker_id': torch.tensor(speaker_idx, dtype=torch.long),
                        'speaker_name': speaker_id,
                        'file_name': wav_file
                    }
                except Exception as e:
                    print(f"Cache loading error for {cache_path}: {e}")
        
        # Load and process audio
        wav_path = osp.join(self.data_root, 'data', speaker_id, wav_file)
        try:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = waveform[0].unsqueeze(0)
            
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
            with torch.no_grad():
                features = self.feature_extractor(waveform)
            
            # Handle length
            n_frames = features.shape[2]
            if n_frames >= self.max_frames:
                start = torch.randint(0, n_frames - self.max_frames + 1, (1,)).item()
                features = features[:, :, start:start + self.max_frames]
            else:
                padding = torch.zeros(1, features.shape[1], self.max_frames - n_frames)
                features = torch.cat([features, padding], dim=2)
            
            # Cache features
            if self.use_cache:
                try:
                    torch.save(features.squeeze(0), cache_path)
                except Exception as e:
                    print(f"Cache saving error for {cache_path}: {e}")
            
            return {
                'features': features.squeeze(0),
                'speaker_id': torch.tensor(speaker_idx, dtype=torch.long),
                'speaker_name': speaker_id,
                'file_name': wav_file
            }
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            return None

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles None values and variable lengths"""
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return {
            'features': torch.tensor([]),
            'speaker_ids': torch.tensor([], dtype=torch.long),
            'speaker_names': [],
            'file_names': []
        }
    
    features = torch.stack([item['features'] for item in batch])
    speaker_ids = torch.stack([item['speaker_id'] for item in batch])
    speaker_names = [item['speaker_name'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    
    return {
        'features': features,
        'speaker_ids': speaker_ids,
        'speaker_names': speaker_names,
        'file_names': file_names
    }
