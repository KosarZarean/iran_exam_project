"""
پیاده‌سازی مدل‌های PyTorch برای داده‌های کنکور ایران
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class ExamMLP(nn.Module):
    """
    شبکه عصبی پیش‌خور برای داده‌های کنکور
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), 
                 num_classes: int = 2, dropout_rate: float = 0.2,
                 activation: str = 'relu', batch_norm: bool = True):
        """
        Parameters:
        -----------
        input_dim : int
            بعد ورودی
        hidden_dims : Tuple[int, ...]
            ابعاد لایه‌های پنهان
        num_classes : int
            تعداد کلاس‌های خروجی
        dropout_rate : float
            نرخ Dropout
        activation : str
            تابع فعال‌ساز
        batch_norm : bool
            استفاده از Batch Normalization
        """
        super(ExamMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = batch_norm
        
        # ساخت لایه‌ها
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # لایه خطی
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch Normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # تابع فعال‌ساز
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'selu':
                layers.append(nn.SELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # لایه خروجی
        if num_classes > 1:
            layers.append(nn.Linear(prev_dim, num_classes))
        else:
            layers.append(nn.Linear(prev_dim, 1))  # برای رگرسیون
        
        self.model = nn.Sequential(*layers)
        
        # تنظیم وزن‌ها
        self._initialize_weights()
    
    def _initialize_weights(self):
        """مقداردهی اولیه وزن‌ها"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """پیش‌برد داده در شبکه"""
        return self.model(x)
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """محاسبه اهمیت ویژگی‌ها با استفاده از وزن‌های لایه اول"""
        if len(feature_names) != self.input_dim:
            raise ValueError(f"تعداد نام ویژگی‌ها ({len(feature_names)}) با بعد ورودی ({self.input_dim}) مطابقت ندارد")
        
        # وزن‌های لایه اول
        first_layer = self.model[0]
        weights = first_layer.weight.data.cpu().numpy()
        
        # محاسبه اهمیت (میانگین قدر مطلق وزن‌های هر ویژگی)
        importance = np.mean(np.abs(weights), axis=0)
        
        # ایجاد دیکشنری
        feature_importance = dict(zip(feature_names, importance))
        
        # مرتب‌سازی بر اساس اهمیت
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        return feature_importance


class ExamTabTransformer(nn.Module):
    """
    پیاده‌سازی TabTransformer برای داده‌های کنکور
    """
    
    def __init__(self, num_categorical: int, num_continuous: int, categories: List[int],
                 num_classes: int = 2, embedding_dim: int = 32, num_heads: int = 4,
                 num_layers: int = 4, transformer_dropout: float = 0.1,
                 mlp_hidden: Tuple[int, ...] = (128, 64), mlp_dropout: float = 0.2):
        """
        Parameters:
        -----------
        num_categorical : int
            تعداد ویژگی‌های دسته‌ای
        num_continuous : int
            تعداد ویژگی‌های عددی
        categories : List[int]
            تعداد مقادیر یکتا برای هر ویژگی دسته‌ای
        num_classes : int
            تعداد کلاس‌های خروجی
        embedding_dim : int
            بعد embedding
        num_heads : int
            تعداد headهای attention
        num_layers : int
            تعداد لایه‌های Transformer
        transformer_dropout : float
            نرخ Dropout در Transformer
        mlp_hidden : Tuple[int, ...]
            ابعاد لایه‌های پنهان MLP
        mlp_dropout : float
            نرخ Dropout در MLP
        """
        super(ExamTabTransformer, self).__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.categories = categories
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Embedding لایه‌ها برای ویژگی‌های دسته‌ای
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) 
            for num_categories in categories
        ])
        
        # لایه‌های Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # لایه‌های MLP برای ویژگی‌های عددی
        if num_continuous > 0:
            self.continuous_projection = nn.Sequential(
                nn.Linear(num_continuous, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(transformer_dropout)
            )
        
        # MLP نهایی
        mlp_input_dim = embedding_dim * num_categorical + (embedding_dim if num_continuous > 0 else 0)
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
        
        if num_classes > 1:
            mlp_layers.append(nn.Linear(prev_dim, num_classes))
        else:
            mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # تنظیم وزن‌ها
        self._initialize_weights()
    
    def _initialize_weights(self):
        """مقداردهی اولیه وزن‌ها"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        """
        پیش‌برد داده در مدل
        
        Parameters:
        -----------
        categorical : torch.Tensor
            ویژگی‌های دسته‌ای با شکل (batch_size, num_categorical)
        continuous : torch.Tensor
            ویژگی‌های عددی با شکل (batch_size, num_continuous)
        """
        # Embedding ویژگی‌های دسته‌ای
        embedded_categorical = []
        for i in range(self.num_categorical):
            emb = self.embeddings[i](categorical[:, i])
            embedded_categorical.append(emb)
        
        # ترکیب embedding‌ها
        embedded_categorical = torch.stack(embedded_categorical, dim=1)  # (batch, num_cat, emb_dim)
        
        # پردازش با Transformer
        transformer_output = self.transformer(embedded_categorical)
        
        # Flatten کردن خروجی Transformer
        transformer_flattened = transformer_output.reshape(
            transformer_output.size(0), -1
        )  # (batch, num_cat * emb_dim)
        
        # پردازش ویژگی‌های عددی
        if self.num_continuous > 0:
            continuous_projected = self.continuous_projection(continuous)
            combined = torch.cat([transformer_flattened, continuous_projected], dim=1)
        else:
            combined = transformer_flattened
        
        # MLP نهایی
        output = self.mlp(combined)
        
        return output
    
    def get_attention_weights(self, categorical: torch.Tensor) -> torch.Tensor:
        """
        دریافت وزن‌های attention برای تفسیرپذیری
        """
        self.eval()
        with torch.no_grad():
            # Embedding ویژگی‌های دسته‌ای
            embedded_categorical = []
            for i in range(self.num_categorical):
                emb = self.embeddings[i](categorical[:, i])
                embedded_categorical.append(emb)
            
            embedded_categorical = torch.stack(embedded_categorical, dim=1)
            
            # محاسبه attention
            attention_weights = []
            for layer in self.transformer.layers:
                # self-attention در هر لایه
                _, attn_weights = layer.self_attn(
                    embedded_categorical, embedded_categorical, embedded_categorical,
                    need_weights=True, average_attn_weights=True
                )
                attention_weights.append(attn_weights)
            
            # میانگین گیری از لایه‌ها
            avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
            
            return avg_attention


class ExamRegressor(nn.Module):
    """
    مدل رگرسیون برای پیش‌بینی رتبه کنکور
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), 
                 dropout_rate: float = 0.2, activation: str = 'relu'):
        super(ExamRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'selu':
                layers.append(nn.SELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # لایه خروجی (یک نورون برای رگرسیون)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """پیش‌برد داده"""
        return self.model(x).squeeze()  # حذف بعد اضافی


class EarlyStopping:
    """
    Early Stopping برای جلوگیری از overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 verbose: bool = True, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> None:
        """
        بررسی Early Stopping
        
        Parameters:
        -----------
        score : float
            امتیاز validation (بیشتر بهتر است)
        model : nn.Module
            مدل PyTorch
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"    🏆 بهترین امتیاز اولیه: {self.best_score:.4f}")
        
        elif score - self.best_score > self.min_delta:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f"    📈 بهبود امتیاز به: {self.best_score:.4f}")
        
        else:
            self.counter += 1
            if self.verbose:
                print(f"    ⏳ عدم بهبود برای {self.counter}/{self.patience} epoch")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("    🛑 توقف زودهنگام فعال شد")
                
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print("    🔄 بارگذاری بهترین وزن‌ها")
    
    def reset(self):
        """بازنشانی Early Stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None