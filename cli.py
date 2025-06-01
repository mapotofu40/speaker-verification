"""
Command-line interface for the speaker verification system
"""
import click
import torch
from pathlib import Path
from config.defaults import BASE_CONFIG
from utils.config import load_config, merge_configs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Speaker Verification System CLI"""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.option('--data-root', required=True, type=click.Path(exists=True), help='Path to dataset root')
@click.option('--metadata-file', required=True, type=click.Path(exists=True), help='Path to metadata file')
@click.option('--utterance-file', required=True, type=click.Path(exists=True), help='Path to utterance list')
def train(config, data_root, metadata_file, utterance_file):
    """Train the speaker verification model"""
    try:
        # Load and merge configurations
        config_dict = BASE_CONFIG.copy()
        if config:
            user_config = load_config(config)
            config_dict = merge_configs(config_dict, user_config)
        
        # Import here to avoid loading everything at startup
        from train import main as train_main
        
        train_main(
            data_root=data_root,
            metadata_file=metadata_file,
            utterance_file=utterance_file,
            config=config_dict
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise click.Abort()

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to model checkpoint')
@click.option('--test-pairs', required=True, type=click.Path(exists=True), help='Path to test pairs file')
@click.option('--wav-dir', required=True, type=click.Path(exists=True), help='Directory containing wav files')
def evaluate(model_path, test_pairs, wav_dir):
    """Evaluate the model on test pairs"""
    try:
        from evaluate import evaluate as evaluate_model
        
        eer = evaluate_model(
            model_path=model_path,
            test_pairs_file=test_pairs,
            wav_dir=wav_dir
        )
        
        click.echo(f"Equal Error Rate: {eer*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.Abort()

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to model checkpoint')
@click.option('--audio1', required=True, type=click.Path(exists=True), help='Path to first audio file')
@click.option('--audio2', required=True, type=click.Path(exists=True), help='Path to second audio file')
@click.option('--threshold', default=0.5, type=float, help='Similarity threshold')
def verify(model_path, audio1, audio2, threshold):
    """Verify if two audio files contain the same speaker"""
    try:
        from verify import verify_pair
        
        result = verify_pair(
            model_path=model_path,
            audio1_path=audio1,
            audio2_path=audio2,
            threshold=threshold
        )
        
        if result:
            click.secho("Same speaker", fg='green')
        else:
            click.secho("Different speakers", fg='red')
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise click.Abort()

if __name__ == '__main__':
    cli()
