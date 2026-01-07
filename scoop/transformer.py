import torch
import torch.nn as nn
import math
import torch.utils.checkpoint

class EstraNet(nn.Module):
    def __init__(self, d_model=128, n_head=8, n_layers=4, num_classes=256):
        super(EstraNet, self).__init__()
        
        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=300, stride=100, padding=100),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
            # 【关键修改】添加这一行，适配 Scoop 的 nll_loss
            nn.LogSoftmax(dim=1)
        )
        self.d_model = d_model

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 1) # (B, L, C)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global Average Pooling
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x