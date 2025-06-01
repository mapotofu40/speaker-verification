"""
Data loading and processing utilities
"""
import os
import os.path as osp
import torch
import torchaudio
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from models.feature_extractor import FeatureExtractor
from utils.augment import AudioAugmentor

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
        use_cache: bool = False,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        musan_path: Optional[str] = None
    ):
        super().__init__()
        
        self.data_root = data_root
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        self.augment = augment
        
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
        
        # Initialize augmentor if needed
        self.augmentor = None
        if augment:
            aug_config = augment_config or {}
            # Add MUSAN path from config or explicit parameter
            if 'musan_path' in aug_config or musan_path:
                aug_config['musan_path'] = musan_path or aug_config.get('musan_path')
            self.augmentor = AudioAugmentor(
                sample_rate=sample_rate,
                **aug_config
            )

    def _get_cache_path(self, speaker_id: str, wav_file: str) -> str:
        uid = f"{speaker_id}_{wav_file}"
        hash_str = hashlib.md5(uid.encode()).hexdigest()
        return osp.join(self.cache_dir, f"{hash_str}.pt")
    
    def _process_audio(self, wav_path: str) -> torch.Tensor:
        """Process audio file into features.
        
        Args:
            wav_path: Path to the audio file
            
        Returns:
            Tensor of shape (1, n_mels, frames) containing the processed features
            
        Raises:
            RuntimeError: If unable to load or process the audio file
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # Apply augmentation if enabled
            if self.augment and self.augmentor is not None:
                try:
                    waveform = self.augmentor(waveform)
                except Exception as e:
                    print(f"Augmentation error for {wav_path}: {str(e)}")
                    # Continue without augmentation
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(waveform)
            
            # Handle length
            n_frames = features.shape[2]
            if n_frames >= self.max_frames:
                # Random crop to max_frames
                start = torch.randint(0, n_frames - self.max_frames + 1, (1,)).item()
                features = features[:, :, start:start + self.max_frames]
            else:
                # Pad with zeros
                padding = torch.zeros(1, features.shape[1], self.max_frames - n_frames)
                features = torch.cat([features, padding], dim=2)
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Error processing audio file {wav_path}: {str(e)}") from e
    
    def __len__(self) -> int:
        return len(self.utterances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dict containing:
                - features: (n_mels, max_frames) tensor of mel spectrogram
                - speaker_id: Integer tensor containing speaker index
                - speaker_name: Original speaker ID string (for reference)
                - file_name: Original audio file name (for reference)
                
        Raises:
            RuntimeError: If unable to load or process the audio file
        """
        speaker_id, wav_file = self.utterances[idx]
        speaker_idx = self.speaker_to_idx[speaker_id]
        
        # Try cache first
        if self.use_cache and not self.augment:  # Don't use cache for augmented data
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
                    print(f"Cache loading error for {cache_path}: {str(e)}")
                    # Fall through to loading from WAV
        
        # Load and process audio
        wav_path = osp.join(self.data_root, 'data', speaker_id, wav_file)
        if not osp.exists(wav_path):
            raise RuntimeError(f"Audio file not found: {wav_path}")
            
        try:
            features = self._process_audio(wav_path)
            
            # Cache features if enabled and not using augmentation
            if self.use_cache and not self.augment:
                try:
                    torch.save(features.squeeze(0), cache_path)
                except Exception as e:
                    print(f"Cache saving error for {cache_path}: {str(e)}")
            
            return {
                'features': features.squeeze(0),
                'speaker_id': torch.tensor(speaker_idx, dtype=torch.long),
                'speaker_name': speaker_id,
                'file_name': wav_file
            }
            
        except Exception as e:
            raise RuntimeError(f"Error processing {wav_path}: {str(e)}") from e


class ValidationPairDataset(Dataset):
    """Dataset for validation with pairs of utterances and their similarity labels"""
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        pair_file: str,
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
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file, sep='\t')
        
        # Load pairs and labels
        self.pairs = []
        with open(pair_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])  # 1 for same speaker, 0 for different
                utt1 = parts[1]
                utt2 = parts[2]
                # Extract speaker_id and wav_file from path
                spk1, wav1 = utt1.split('/')
                spk2, wav2 = utt2.split('/')
                self.pairs.append(((spk1, wav1), (spk2, wav2), label))
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor if feature_extractor else FeatureExtractor()
    
    def _get_cache_path(self, speaker_id: str, wav_file: str) -> str:
        uid = f"{speaker_id}_{wav_file}"
        hash_str = hashlib.md5(uid.encode()).hexdigest()
        return osp.join(self.cache_dir, f"{hash_str}.pt")
    
    def _process_audio(self, wav_path: str) -> torch.Tensor:
        """Process audio file into features"""
        waveform, sr = torchaudio.load(wav_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(waveform)
        
        # Handle length
        n_frames = features.shape[2]
        if n_frames >= self.max_frames:
            # For validation, take center frames
            start = (n_frames - self.max_frames) // 2
            features = features[:, :, start:start + self.max_frames]
        else:
            padding = torch.zeros(1, features.shape[1], self.max_frames - n_frames)
            features = torch.cat([features, padding], dim=2)
        
        return features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        (spk1, wav1), (spk2, wav2), label = self.pairs[idx]
        
        # Process first utterance
        features1 = None
        if self.use_cache:
            cache_path1 = self._get_cache_path(spk1, wav1)
            if osp.exists(cache_path1):
                try:
                    features1 = torch.load(cache_path1)
                except Exception:
                    pass
        
        if features1 is None:
            wav_path1 = osp.join(self.data_root, 'data', spk1, wav1)
            features1 = self._process_audio(wav_path1)
            if self.use_cache:
                torch.save(features1.squeeze(0), cache_path1)
        
        # Process second utterance
        features2 = None
        if self.use_cache:
            cache_path2 = self._get_cache_path(spk2, wav2)
            if osp.exists(cache_path2):
                try:
                    features2 = torch.load(cache_path2)
                except Exception:
                    pass
        
        if features2 is None:
            wav_path2 = osp.join(self.data_root, 'data', spk2, wav2)
            features2 = self._process_audio(wav_path2)
            if self.use_cache:
                torch.save(features2.squeeze(0), cache_path2)
        
        return {
            'features1': features1.squeeze(0),
            'features2': features2.squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'spk1': spk1,
            'spk2': spk2,
            'wav1': wav1,
            'wav2': wav2
        }
    
    def __len__(self) -> int:
        return len(self.pairs)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles None values and variable lengths"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    # Assume all items have the same keys
    output = {}
    keys = batch[0].keys()
    
    for key in keys:
        # Only include tensor fields in the output
        if isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch])
        # Skip non-tensor fields as they're not needed for training
    
    return output
