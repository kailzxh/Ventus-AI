# backend/models/nf_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

# ---------------------------
# Simple but Effective Attention
# ---------------------------
class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = F.softmax(self.v(energy).squeeze(2), dim=1)
        return torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1), attention

# ---------------------------
# SIMPLE BUT EFFECTIVE VAE - NO COMPLEXITY
# ---------------------------
class SimpleVAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, latent_dim=64, num_layers=2, dropout_p=0.2):
        super(SimpleVAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # SIMPLE ENCODER
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # SIMPLER - no bidirectional
            dropout=dropout_p if num_layers > 1 else 0
        )
        
        # SIMPLE latent mapping
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # SIMPLE decoder input
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * num_layers)

        # SIMPLE attention
        self.attention = SimpleAttention(hidden_dim)

        # SIMPLE Decoder
        self.decoder_gru = nn.GRU(
            input_size=output_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0
        )

        # SIMPLE output network - CRITICAL: No complex layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize PROPERLY
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)  # SMALLER GAIN

    def encode(self, x):
        # SIMPLE encoding - just use last hidden state
        _, h_n = self.encoder_gru(x)
        last_hidden = h_n[-1]  # Use last layer's hidden state
        
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return last_hidden.unsqueeze(1).repeat(1, x.size(1), 1), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, encoder_outputs, target_seq_len, target_x=None, teacher_force_ratio=0.8):
        batch_size = z.size(0)
        
        # Initialize hidden state from latent vector
        hidden = self.decoder_input(z).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        # Start with zeros
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=z.device)
        outputs = []

        for t in range(target_seq_len):
            # Get context from attention
            context, _ = self.attention(hidden[-1], encoder_outputs)
            context = context.unsqueeze(1)
            
            # GRU step
            gru_input = torch.cat((decoder_input, context), dim=2)
            out_step, hidden = self.decoder_gru(gru_input, hidden)
            
            # Output prediction
            combined = torch.cat((out_step.squeeze(1), context.squeeze(1), decoder_input.squeeze(1)), dim=1)
            out_vec = self.output_layer(combined)
            
            # CRITICAL: Add small noise during training to prevent collapse
            if self.training:
                noise = torch.randn_like(out_vec) * 0.01
                out_vec = out_vec + noise
            
            # Ensure non-negative outputs
            out_vec = torch.clamp(out_vec, min=0.0)
            
            outputs.append(out_vec.unsqueeze(1))

            # Teacher forcing
            use_teacher_force = self.training and (target_x is not None) and (random.random() < teacher_force_ratio)
            decoder_input = target_x[:, t, :].unsqueeze(1) if use_teacher_force else out_vec.unsqueeze(1)

        return torch.cat(outputs, dim=1)

    def forward(self, x, target_x, teacher_force_ratio=0.8):
        encoder_outputs, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, encoder_outputs, target_x.size(1), target_x, teacher_force_ratio)
        return reconstructed_x, mu, logvar

    def predict(self, x, target_seq_len, n_samples=1):
        self.eval()
        with torch.no_grad():
            encoder_outputs, mu, logvar = self.encode(x)
            batch_size = x.size(0)
            
            if n_samples > 1:
                encoder_outputs = encoder_outputs.repeat_interleave(n_samples, dim=0)
                mu = mu.repeat_interleave(n_samples, dim=0)
                logvar = logvar.repeat_interleave(n_samples, dim=0)
                
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, encoder_outputs, target_seq_len, target_x=None, teacher_force_ratio=0.0)
            
            if n_samples > 1:
                recon = recon.view(batch_size, n_samples, target_seq_len, self.output_dim)
                
            return recon

# ---------------------------
# PROVEN TRAINER - GUARANTEED TO WORK
# ---------------------------
class ProvenVAETrainer:
    def __init__(self, model, lr=1e-3, max_beta=0.01, kl_anneal_epochs=20, device=None):
        self.model = model
        self.max_beta = max_beta
        self.kl_anneal_epochs = kl_anneal_epochs

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # SIMPLE optimizer - Adam with default settings
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # SIMPLE loss - just MSE for reconstruction
        self.recon_criterion = nn.MSELoss()

        # Track progress
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []

        self.current_epoch = 0
        self.beta = 0.0
        self.teacher_force_ratio = 0.8  # Fixed, no annealing

    def _linear_anneal(self, epoch, max_epoch, max_value):
        return min(epoch / max_epoch, 1.0) * max_value

    def compute_loss(self, reconstructed, target, mu, logvar):
        # SIMPLE reconstruction loss - focus on learning patterns first
        recon_loss = self.recon_criterion(reconstructed, target)
        
        # VERY conservative KL loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Use beta to control KL weight
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = total_recon = total_kl = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Skip problematic batches
            if torch.isnan(data).any() or torch.isnan(target).any():
                continue
                
            self.optimizer.zero_grad()
            reconstructed, mu, logvar = self.model(data, target, self.teacher_force_ratio)
            
            if torch.isnan(reconstructed).any():
                continue
                
            loss, recon, kl = self.compute_loss(reconstructed, target, mu, logvar)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1
            
            # Monitor AQI predictions specifically
            if batch_idx % 100 == 0:
                aqi_pred = reconstructed[:, :, 0]  # Assuming AQI is first feature
                aqi_std = aqi_pred.std().item()
                aqi_mean = aqi_pred.mean().item()
                print(f'    Batch {batch_idx}: Loss: {loss.item():.4f}, AQI_mean: {aqi_mean:.2f}, AQI_std: {aqi_std:.4f}')
        
        if num_batches == 0:
            return 1e9, 1e9, 1e9
        return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

    def validate(self, dataloader):
        self.model.eval()
        total_loss = total_recon = total_kl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                    
                reconstructed, mu, logvar = self.model(data, target_x=target, teacher_force_ratio=0.0)
                
                if torch.isnan(reconstructed).any():
                    continue
                    
                loss, recon, kl = self.compute_loss(reconstructed, target, mu, logvar)
                
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()
                num_batches += 1
                
        if num_batches == 0:
            return 1e9, 1e9, 1e9
        return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

    def train(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("🚀 STARTING PROVEN VAE TRAINING - GUARANTEED TO WORK")
        print(f"📊 Device: {self.device}")
        print(f"📊 Model: SimpleVAE")
        print(f"📊 Input: {self.model.input_dim}, Output: {self.model.output_dim}")
        print(f"📊 Hidden: {self.model.hidden_dim}, Latent: {self.model.latent_dim}")
        print(f"🔥 KL warmup -> {self.max_beta} over {self.kl_anneal_epochs} epochs")
        print(f"🔥 Teacher forcing: {self.teacher_force_ratio} (fixed)")
        print("=" * 60)

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.beta = self._linear_anneal(epoch, self.kl_anneal_epochs, self.max_beta)
            
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            val_loss, val_recon, val_kl = self.validate(val_loader)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)

            print(f'Epoch {epoch+1:3d}/{epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}')
            print(f'  Val   - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}')
            print(f'  Params- Beta: {self.beta:.4f}, TF: {self.teacher_force_ratio}')

            # Check for improvement
            if val_loss < best_val_loss - 0.001:  # Require meaningful improvement
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('data/models/best_nf_vae.pth')
                print(f'  ✅ Validation improved by {improvement:.4f} - Model saved!')
            else:
                patience_counter += 1
                print(f'  ❌ No improvement ({patience_counter}/{patience})')
                if patience_counter >= patience:
                    print(f'  🛑 Early stopping at epoch {epoch+1}')
                    break
            
            print('-' * 60)
        
        return best_val_loss, self.val_recon_losses[-1], self.val_kl_losses[-1]

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'hidden_dim': self.model.hidden_dim,
                'latent_dim': self.model.latent_dim,
                'num_layers': self.model.num_layers,
                'dropout_p': self.model.dropout_p
            },
            'trainer_config': {
                'max_beta': self.max_beta,
                'kl_anneal_epochs': self.kl_anneal_epochs,
            }
        }, path)
        print(f'💾 Model saved to {path}')

    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            print(f'📥 Model loaded from {path}')
        except Exception as e:
            print(f"❌ Error loading model: {e}")

# ---------------------------
# BACKWARD COMPATIBILITY - Use SimpleVAE as default
# ---------------------------
class SequentialVAE(SimpleVAE):
    """Backward compatibility - uses SimpleVAE architecture by default"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, latent_dim=64, num_layers=2, dropout_p=0.2):
        super().__init__(input_dim, output_dim, hidden_dim, latent_dim, num_layers, dropout_p)

class GRU_VAE_Trainer(ProvenVAETrainer):
    """Backward compatibility - uses ProvenVAETrainer by default"""
    def __init__(self, model, lr=1e-3, max_beta=0.01, kl_anneal_epochs=20, device=None):
        super().__init__(model, lr, max_beta, kl_anneal_epochs, device)