"""Inference model retained from the original ResNet_Trans/train_custom.py."""
import math
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def load_pretrained_weights(model_name, pretrained_dir):
    """사전학습된 가중치를 로드하는 함수"""
    weights_path = os.path.join(pretrained_dir, f"{model_name}.pth")
    if os.path.exists(weights_path):
        return torch.load(weights_path, weights_only=True)
    return None

def save_pretrained_weights(model, model_name, pretrained_dir):
    """사전학습된 가중치를 저장하는 함수"""
    os.makedirs(pretrained_dir, exist_ok=True)
    weights_path = os.path.join(pretrained_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), weights_path)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, features):
        # features: [seq_len, batch_size, hidden_size]
        seq_len, batch_size, hidden_size = features.size()
        
        # Calculate attention weights
        attn_weights = self.attention(features)  # [seq_len, batch_size, 1]
        attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=0)  # [seq_len, batch_size]
        
        # Apply attention weights
        weighted_features = features * attn_weights.unsqueeze(-1)  # [seq_len, batch_size, hidden_size]
        pooled = torch.sum(weighted_features, dim=0)  # [batch_size, hidden_size]
        
        return pooled

class ResNetTransformerImproved(nn.Module):
    """개선된 ResNet + Transformer Encoder 모델"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 프레임 샘플링 설정
        self.max_frames = getattr(cfg.MODEL.SEQUENCE, 'MAX_FRAMES', 900)
        self.frame_sampling = getattr(cfg.MODEL.SEQUENCE, 'FRAME_SAMPLING', 'uniform')
        self.use_global_pooling = getattr(cfg.MODEL.SEQUENCE, 'USE_GLOBAL_POOLING', True)
        self.pooling_type = getattr(cfg.MODEL.SEQUENCE, 'POOLING_TYPE', 'attention')

        # ResNet 백본
        pretrained_dir = cfg.MODEL.BACKBONE.PRETRAINED_DIR
        if cfg.MODEL.BACKBONE.PRETRAINED:
            weights = load_pretrained_weights("resnet18_imagenet", pretrained_dir)
            if weights is not None:
                self.backbone = models.resnet18(weights=None)
                self.backbone.load_state_dict(weights)
            else:
                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                save_pretrained_weights(self.backbone, "resnet18", pretrained_dir)
        else:
            self.backbone = models.resnet18(weights=None)
        
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Transformer Encoder (개선된 구조)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=cfg.MODEL.TRANSFORMER.NUM_HEADS,
            dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
            batch_first=False  # 시퀀스가 첫 번째 차원
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.TRANSFORMER.NUM_LAYERS)
        self.pos_encoder = PositionalEncoding(num_features, max_len=self.max_frames)
        
        # 풀링 레이어 설정
        if self.use_global_pooling:
            if self.pooling_type == "attention":
                self.pooling = AttentionPooling(num_features)
            elif self.pooling_type == "mean":
                self.pooling = lambda x: torch.mean(x, dim=0)
            elif self.pooling_type == "max":
                self.pooling = lambda x: torch.max(x, dim=0)[0]
            else:  # "last"
                self.pooling = lambda x: x[-1]
        else:
            self.pooling = lambda x: x[-1]
        
        # 향상된 분류기
        self.classifier = nn.Sequential(
            nn.Linear(num_features, cfg.MODEL.CLASSIFIER.HIDDEN_SIZE),
            nn.LayerNorm(cfg.MODEL.CLASSIFIER.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.CLASSIFIER.DROPOUT),
            nn.Linear(cfg.MODEL.CLASSIFIER.HIDDEN_SIZE, cfg.MODEL.CLASSIFIER.HIDDEN_SIZE // 2),
            nn.LayerNorm(cfg.MODEL.CLASSIFIER.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.CLASSIFIER.DROPOUT / 2),
            nn.Linear(cfg.MODEL.CLASSIFIER.HIDDEN_SIZE // 2, cfg.NUM_CLASSES)
        )

    def sample_frames(self, x):
        """프레임 샘플링 함수"""
        batch_size, seq_len, c, h, w = x.size()
        
        if seq_len <= self.max_frames:
            return x
        
        if self.frame_sampling == 'uniform':
            # 균등 샘플링
            indices = torch.linspace(0, seq_len - 1, self.max_frames, dtype=torch.long)
            return x[:, indices]
        elif self.frame_sampling == 'random':
            # 랜덤 샘플링 (학습 시에만)
            if self.training:
                indices = torch.randperm(seq_len)[:self.max_frames].sort()[0]
                return x[:, indices]
            else:
                indices = torch.linspace(0, seq_len - 1, self.max_frames, dtype=torch.long)
                return x[:, indices]
        else:  # 'center'
            # 중앙 부분 샘플링
            start_idx = (seq_len - self.max_frames) // 2
            end_idx = start_idx + self.max_frames
            return x[:, start_idx:end_idx]

    def forward(self, x):
        # 입력 x의 크기: [batch_size, seq_len, 3, 32, 64]
        batch_size, seq_len, c, h, w = x.size()
        
        # 프레임 샘플링 적용
        x = self.sample_frames(x)
        seq_len = x.size(1)  # 샘플링 후 길이 업데이트
        
        # 배치와 시퀀스 차원 병합하여 ResNet 처리
        x = x.view(batch_size * seq_len, c, h, w)
        
        # ResNet 특징 추출
        features = self.backbone(x)  # [batch_size * seq_len, 512]
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
        
        # Transformer 처리를 위한 차원 변경
        features = features.permute(1, 0, 2)  # [seq_len, batch_size, 512]
        
        # Positional encoding 적용
        features = self.pos_encoder(features)
        
        # Transformer 인코딩
        features = self.transformer(features)  # [seq_len, batch_size, 512]
        
        # 풀링 적용
        pooled_features = self.pooling(features)  # [batch_size, 512]
        
        # 분류
        logits = self.classifier(pooled_features)
        return logits
