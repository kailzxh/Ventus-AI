# backend/models/nf_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedNFVAE(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, latent_dim=32,  # Increased dimensions for more features
                 sequence_length=24, num_pollutants=7):
        super(EnhancedNFVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.num_pollutants = num_pollutants
        
        # Enhanced Encoder with more capacity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * sequence_length, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # Latent space
        encoder_out_dim = hidden_dim // 2
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)
        
        # Enhanced Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_pollutants * sequence_length),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def encode(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        batch_size = z.shape[0]
        reconstructed = self.decoder(z)
        return reconstructed.reshape(batch_size, self.sequence_length, self.num_pollutants)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def predict(self, x):
        """Prediction without sampling for inference"""
        with torch.no_grad():
            mu, logvar = self.encode(x)
            reconstructed = self.decode(mu)
            return reconstructed

class ComprehensiveNFVAETrainer:
    def __init__(self, model, aqi_weight=2.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.aqi_weight = aqi_weight
        
        # Optimizer with different learning rates for encoder/decoder
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': 0.0003},
            {'params': decoder_params, 'lr': 0.0005}
        ], weight_decay=1e-5, betas=(0.9, 0.999))
        
        # Multiple loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.5, min_lr=1e-6
        )
        
        # Track multiple metrics
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.train_aqi_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        self.val_aqi_losses = []
    
    def compute_loss(self, reconstructed, target, mu, logvar, beta=0.1):
        # Combined reconstruction loss (MSE + MAE)
        mse_loss = self.mse_loss(reconstructed, target)
        mae_loss = self.mae_loss(reconstructed, target)
        recon_loss = 0.7 * mse_loss + 0.3 * mae_loss
        
        # AQI-specific loss (last feature)
        aqi_recon = reconstructed[:, :, -1]
        aqi_target = target[:, :, -1]
        aqi_mse = self.mse_loss(aqi_recon, aqi_target)
        aqi_mae = self.mae_loss(aqi_recon, aqi_target)
        aqi_loss = 0.7 * aqi_mse + 0.3 * aqi_mae
        
        # Combined loss with higher weight for AQI
        weighted_recon_loss = 0.6 * recon_loss + 0.4 * self.aqi_weight * aqi_loss
        
        # KL divergence with numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)
        kl_loss = torch.clamp(kl_loss, min=0, max=1e6)
        
        # Monitor for NaN
        if torch.isnan(weighted_recon_loss) or torch.isnan(kl_loss):
            print(f"⚠️  NaN detected - Recon: {weighted_recon_loss.item()}, KL: {kl_loss.item()}")
            return (torch.tensor(1.0, device=self.device), 
                    torch.tensor(1.0, device=self.device), 
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0, device=self.device))
        
        # Total loss
        total_loss = weighted_recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, aqi_loss
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_aqi_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Skip batches with NaN
            if torch.isnan(data).any() or torch.isnan(target).any():
                print(f"⚠️  Skipping batch {batch_idx} due to NaN values")
                continue
            
            self.optimizer.zero_grad()
            
            reconstructed, mu, logvar = self.model(data)
            
            # Check for NaN in model outputs
            if torch.isnan(reconstructed).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
                print(f"⚠️  Skipping batch {batch_idx} due to NaN in model outputs")
                continue
            
            loss, recon_loss, kl_loss, aqi_loss = self.compute_loss(reconstructed, target, mu, logvar)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                print(f"⚠️  Skipping batch {batch_idx} due to NaN loss")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_aqi_loss += aqi_loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'    Batch {batch_idx:3d}, Loss: {loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, '
                      f'AQI: {aqi_loss.item():.4f}')
        
        if num_batches == 0:
            return 1.0, 1.0, 0.0, 1.0
        
        return (total_loss / num_batches, 
                total_recon_loss / num_batches, 
                total_kl_loss / num_batches,
                total_aqi_loss / num_batches)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_aqi_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Skip batches with NaN
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                
                reconstructed, mu, logvar = self.model(data)
                
                # Skip if model outputs contain NaN
                if torch.isnan(reconstructed).any():
                    continue
                
                loss, recon_loss, kl_loss, aqi_loss = self.compute_loss(reconstructed, target, mu, logvar)
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    continue
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_aqi_loss += aqi_loss.item()
                num_batches += 1
        
        if num_batches == 0:
            return 1.0, 1.0, 0.0, 1.0
        
        return (total_loss / num_batches, 
                total_recon_loss / num_batches, 
                total_kl_loss / num_batches,
                total_aqi_loss / num_batches)
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🏃 Starting training for {epochs} epochs...")
        print(f"📊 Device: {self.device}")
        print(f"📊 Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        print(f"📊 Model: Enhanced NF-VAE with comprehensive features")
        print(f"📊 Input dimension: {self.model.input_dim}")
        print(f"📊 AQI Weight: {self.aqi_weight}")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_recon, train_kl, train_aqi = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_recon, val_kl, val_aqi = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.train_aqi_losses.append(train_aqi)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)
            self.val_aqi_losses.append(val_aqi)
            
            self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1:3d}/{epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}, AQI: {train_aqi:.4f}')
            print(f'  Val   - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, AQI: {val_aqi:.4f}')
            print(f'  LR: {current_lr:.6f}')
            
            # Check for improvement
            if val_loss < best_val_loss - 0.0001:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('data/models/best_nf_vae.pth')
                print(f'  ✅ Validation loss improved by {improvement:.4f}')
                print('  💾 Model saved!')
            else:
                patience_counter += 1
                print(f'  ❌ No improvement ({patience_counter}/{patience})')
                if patience_counter >= patience:
                    print(f'  🛑 Early stopping at epoch {epoch+1}')
                    break
            
            print('-' * 60)
        
        # Return final validation metrics (3 values for backward compatibility)
        return best_val_loss, self.val_recon_losses[-1], self.val_kl_losses[-1]
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_recon_losses': self.train_recon_losses,
            'train_kl_losses': self.train_kl_losses,
            'train_aqi_losses': self.train_aqi_losses,
            'val_recon_losses': self.val_recon_losses,
            'val_kl_losses': self.val_kl_losses,
            'val_aqi_losses': self.val_aqi_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'latent_dim': self.model.latent_dim,
                'sequence_length': self.model.sequence_length,
                'num_pollutants': self.model.num_pollutants
            },
            'trainer_config': {
                'aqi_weight': self.aqi_weight
            }
        }, path)
        print(f'💾 Model saved to {path}')
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_recon_losses = checkpoint.get('train_recon_losses', [])
        self.train_kl_losses = checkpoint.get('train_kl_losses', [])
        self.train_aqi_losses = checkpoint.get('train_aqi_losses', [])
        self.val_recon_losses = checkpoint.get('val_recon_losses', [])
        self.val_kl_losses = checkpoint.get('val_kl_losses', [])
        self.val_aqi_losses = checkpoint.get('val_aqi_losses', [])
        self.aqi_weight = checkpoint.get('trainer_config', {}).get('aqi_weight', 2.0)
        print(f'📥 Model loaded from {path}')

# Backward compatibility
class AQIFocusedNFVAETrainer(ComprehensiveNFVAETrainer):
    def __init__(self, model, aqi_weight=2.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, aqi_weight, device)

class StableNFVAE(EnhancedNFVAE):
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=16,
                 sequence_length=24, num_pollutants=7):
        super().__init__(input_dim, hidden_dim, latent_dim, sequence_length, num_pollutants)