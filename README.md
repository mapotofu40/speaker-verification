# Speaker Verification System

This project implements an ECAPA-TDNN based speaker verification system using PyTorch.

## Features

- ECAPA-TDNN architecture for speaker embedding extraction
- Angular Additive Margin (AAM) Softmax for speaker classification
- Feature caching for faster training
- Robust data loading and augmentation pipeline
- Checkpointing and training resumption
- Evaluation and inference scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speaker-verification.git
cd speaker-verification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py               # Configuration settings
├── models/                # Model architecture
│   ├── ecapa_tdnn.py     # ECAPA-TDNN implementation
│   ├── feature_extractor.py  # Audio feature extraction
│   └── modules.py        # Neural network modules
├── utils/                # Utility functions
│   ├── data.py          # Dataset and data loading
│   ├── metrics.py       # Evaluation metrics
│   └── training.py      # Training functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── verify.py            # Inference script

```

## Usage

### Training

```bash
python train.py --data_root /path/to/data --metadata_file metadata.tsv --utterance_file utterances.txt
```

### Evaluation

```bash
python evaluate.py --model_path /path/to/model.pth --test_pairs test_pairs.txt --wav_dir /path/to/wavs
```

### Speaker Verification

```bash
python verify.py --model_path /path/to/model.pth --audio1 speaker1.wav --audio2 speaker2.wav
```

## Configuration

Key parameters can be modified in `config.py`:

- Audio processing parameters (sample rate, window size, etc.)
- Model architecture (embedding dimension, number of channels, etc.)
- Training parameters (batch size, learning rate, etc.)

## Model Architecture

The system uses the ECAPA-TDNN architecture with the following components:

1. Feature Extraction:
   - Mel-frequency cepstral coefficients (MFCCs)
   - Optional pre-emphasis filtering

2. ECAPA-TDNN:
   - Multi-scale temporal context aggregation
   - Squeeze-and-excitation attention
   - Res2Net-style feature processing

3. Speaker Classification:
   - Angular Additive Margin Softmax
   - Cosine similarity scoring

## Dataset Format

The dataset should be organized as follows:

```
data_root/
├── data/
│   ├── speaker1/
│   │   ├── utterance1.wav
│   │   └── utterance2.wav
│   └── speaker2/
│       ├── utterance1.wav
│       └── utterance2.wav
├── metadata.tsv
└── utterances.txt
```

- `metadata.tsv`: Tab-separated file with speaker information
- `utterances.txt`: Tab-separated file listing speaker_id and wav_file paths

## License

MIT License
