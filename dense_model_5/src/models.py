import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

class XYErrorNet(nn.Module):
    """
    Neural Network for predicting X and Y forward kinematics errors.
    Uses only joint angles as input.
    """
    def __init__(self, dropout_rate=0.25):
        super(XYErrorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 2000),  # Input: 6 joint angles
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(500, 2)  # Output: delta_x, delta_y
        )
    
    def forward(self, x):
        return self.model(x)


class ZErrorNet(nn.Module):
    """
    Neural Network for predicting Z forward kinematics error.
    Uses joint angles AND joint torques as input.
    """
    def __init__(self, dropout_rate=0.25):
        super(ZErrorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 2000),  # Input: 6 joint angles + 6 joint torques
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(500, 1)  # Output: delta_z
        )
    
    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_model(model, X_train, y_train, X_val, y_val, scaler_y,
                batch_size=64, epochs=150, lr=0.001, patience=15,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                model_name="model"):
    """
    Train a neural network model with the specified parameters
    
    Args:
        model: Neural network model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        scaler_y: Scaler for target values
        batch_size: Training batch size
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to run on
        model_name: Name for progress messages
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Calculate number of batches
    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Create random indices for shuffling
        indices = torch.randperm(n_samples)
        
        # Process batches
        for i in range(n_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / n_samples
        history['train_loss'].append(avg_train_loss)
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            history['val_loss'].append(val_loss)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'[{model_name}] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {current_lr}')
            
            # Calculate metrics on validation set (in original scale)
            y_pred_scaled = val_outputs.cpu().numpy()
            
            # For inverse transform, handle dimensionality correctly
            if y_pred_scaled.shape[1] != scaler_y.scale_.shape[0]:
                # Create a dummy array with correct shape for inverse transform
                dummy_pred = np.zeros((len(y_pred_scaled), scaler_y.scale_.shape[0]))
                dummy_pred[:, :y_pred_scaled.shape[1]] = y_pred_scaled
                y_pred = scaler_y.inverse_transform(dummy_pred)[:, :y_pred_scaled.shape[1]]
            else:
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                
            y_true = y_val.cpu().numpy()
            if y_true.shape[1] != scaler_y.scale_.shape[0]:
                dummy_true = np.zeros((len(y_true), scaler_y.scale_.shape[0]))
                dummy_true[:, :y_true.shape[1]] = y_true
                y_true = scaler_y.inverse_transform(dummy_true)[:, :y_true.shape[1]]
            else:
                y_true = scaler_y.inverse_transform(y_true)
            
            # Calculate RMSE
            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            
            print(f'[{model_name}] Validation RMSE: {rmse:.8f} meters')
        
        # Check for early stopping
        if early_stopping(val_loss):
            print(f"[{model_name}] Early stopping triggered at epoch {epoch+1}")
            break
    
    # Calculate total training time
    train_time = time.time() - start_time
    print(f'[{model_name}] Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)')
    
    return model, history