"""
Training utilities
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingStats:
    """Tracks training statistics"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
    def get_stats(self) -> Dict[str, Any]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_dir: str = "./checkpoints",
    resume_from: Optional[str] = None,
    checkpoint_interval: int = 5
) -> nn.Module:
    """
    Train the speaker verification model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        resume_from: Optional path to checkpoint to resume from
        checkpoint_interval: Save checkpoint every N epochs
        
    Returns:
        Trained model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training
    try:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        stats = TrainingStats()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            try:
                logging.info(f"Resuming from checkpoint: {resume_from}")
                checkpoint = torch.load(resume_from)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                stats.best_val_loss = checkpoint['best_val_loss']
                logging.info(f"Resumed from epoch {start_epoch}, best validation loss: {stats.best_val_loss:.4f}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                raise
        
        # Training loop
        start_time = time.time()
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
            
            for batch_idx, batch in enumerate(train_iterator):
                try:
                    features = batch['features'].to(device)
                    speaker_ids = batch['speaker_ids'].to(device)
                    
                    if features.shape[0] == 0:
                        continue
                    
                    logits = model(features, speaker_ids)
                    loss = criterion(logits, speaker_ids)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # Gradient clipping
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_iterator.set_postfix(loss=loss.item())
                    
                except Exception as e:
                    logging.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")
                    for batch_idx, batch in enumerate(val_iterator):
                        try:
                            features = batch['features'].to(device)
                            speaker_ids = batch['speaker_ids'].to(device)
                            
                            logits = model(features, speaker_ids)
                            loss = criterion(logits, speaker_ids)
                            val_loss += loss.item()
                            
                        except Exception as e:
                            logging.error(f"Error in validation batch {batch_idx}: {e}")
                            continue
                    
                    val_loss /= len(val_loader)
                    
                    # Save best model
                    if val_loss < stats.best_val_loss:
                        stats.best_val_loss = val_loss
                        best_model_path = checkpoint_dir / 'best_model.pth'
                        torch.save(model.state_dict(), best_model_path)
                        logging.info(f"New best model saved to {best_model_path}")
            
            # Update learning rate
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # Update statistics
            stats.update(epoch, train_loss, val_loss, current_lr)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.1f}s - "
                f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                f"LR: {current_lr:.6f}"
            )
            
            # Save periodic checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'stats': stats.get_stats()
                }, checkpoint_path)
                logging.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Training complete
        total_time = time.time() - start_time
        logging.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logging.info(f"Best validation loss: {stats.best_val_loss:.4f} at epoch {stats.best_epoch}")
        
        # Load best model if validation was performed
        if val_loader is not None:
            best_model_path = checkpoint_dir / 'best_model.pth'
            if best_model_path.exists():
                model.load_state_dict(torch.load(best_model_path))
                logging.info(f"Loaded best model from {best_model_path}")
        
        return model
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
