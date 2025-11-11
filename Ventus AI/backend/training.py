# backend/training.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class AQIModelTrainer:
    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, learning_rate=1e-3):
        """Train the model"""
        print(f"🚀 Training {self.model_name} for {epochs} epochs...")
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    # For NF-VAE
                    x_recon, recon_loss, kl_loss = self.model(batch_X)
                    loss = recon_loss + 0.1 * kl_loss
                else:
                    # For standard models
                    predictions = self.model(batch_X)
                    loss = torch.nn.functional.mse_loss(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if hasattr(self.model, 'forward'):
                        x_recon, recon_loss, kl_loss = self.model(batch_X)
                        val_loss += (recon_loss + 0.1 * kl_loss).item()
                    else:
                        predictions = self.model(batch_X)
                        val_loss += torch.nn.functional.mse_loss(predictions, batch_y).item()
            
            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"data/models/best_{self.model_name}.pth")
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        print(f"✅ Training completed! Best validation loss: {self.best_val_loss:.4f}")
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        self.model.eval()
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                if hasattr(self.model, 'forward'):
                    x_recon, _, _ = self.model(batch_X)
                    predictions.extend(x_recon.cpu().numpy())
                else:
                    preds = self.model(batch_X)
                    predictions.extend(preds.cpu().numpy())
                
                targets.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae, 
            'R2': r2,
            'MSE': mse
        }
        
        print("📊 Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics, predictions, targets
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"📥 Model loaded from {filepath}")

def prepare_training_data(df, feature_cols, target_col='AQI', test_size=0.2, val_size=0.2):
    """Prepare data for training"""
    from sklearn.model_selection import train_test_split
    
    # Ensure we have the target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Get features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, shuffle=False
    )
    
    print(f"📊 Data splits: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test