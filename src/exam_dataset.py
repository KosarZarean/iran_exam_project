"""
کلاس‌های Dataset برای داده‌های کنکور ایران
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class ExamTorchDataset(Dataset):
    """
    PyTorch Dataset برای مدل‌های سنتی (MLP، رگرسیون و مدل‌های sklearn)
    """
    
    def __init__(self, X, y):
        """
        Parameters:
        -----------
        X : array-like
            ویژگی‌ها
        y : array-like
            برچسب‌ها
        """
        self.X = torch.FloatTensor(X)
        
        # تشخیص نوع برچسب (طبقه‌بندی یا رگرسیون)
        if len(np.unique(y)) < 100 and y.dtype in [np.int32, np.int64]:
            # طبقه‌بندی
            self.y = torch.LongTensor(y)
            self.task_type = 'classification'
        else:
            # رگرسیون
            self.y = torch.FloatTensor(y)
            self.task_type = 'regression'
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExamTabTransformerDataset(Dataset):
    """
    PyTorch Dataset برای مدل TabTransformer
    """
    
    def __init__(self, X_cat, X_cont, y):
        """
        Parameters:
        -----------
        X_cat : array-like
            ویژگی‌های دسته‌ای (categorical)
        X_cont : array-like
            ویژگی‌های عددی (continuous)
        y : array-like
            برچسب‌ها
        """
        self.X_cat = torch.LongTensor(X_cat)
        self.X_cont = torch.FloatTensor(X_cont)
        
        # تشخیص نوع برچسب
        if len(np.unique(y)) < 100 and y.dtype in [np.int32, np.int64]:
            self.y = torch.LongTensor(y)
            self.task_type = 'classification'
        else:
            self.y = torch.FloatTensor(y)
            self.task_type = 'regression'
    
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


class ExamInferenceDataset(Dataset):
    """
    PyTorch Dataset برای پیش‌بینی (inference)
    بدون برچسب
    """
    
    def __init__(self, X, is_tabtransformer=False):
        """
        Parameters:
        -----------
        X : array-like or tuple
            ویژگی‌ها
        is_tabtransformer : bool
            آیا داده برای TabTransformer است؟
        """
        self.is_tabtransformer = is_tabtransformer
        
        if is_tabtransformer:
            # برای TabTransformer: (X_cat, X_cont)
            X_cat, X_cont = X
            self.X_cat = torch.LongTensor(X_cat)
            self.X_cont = torch.FloatTensor(X_cont)
        else:
            # برای مدل‌های عادی
            self.X = torch.FloatTensor(X)
    
    def __len__(self):
        if self.is_tabtransformer:
            return len(self.X_cat)
        else:
            return len(self.X)
    
    def __getitem__(self, idx):
        if self.is_tabtransformer:
            return self.X_cat[idx], self.X_cont[idx]
        else:
            return self.X[idx]