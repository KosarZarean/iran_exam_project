"""
آموزش مدل‌های یادگیری ماشین برای داده‌های کنکور ایران
"""

import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from exam_models import ExamMLP, ExamTabTransformer, ExamRegressor, EarlyStopping
from exam_dataset import ExamTorchDataset, ExamTabTransformerDataset


class ExamModelTrainer:
    """
    کلاس جامع برای آموزش و تنظیم هایپرپارامتر مدل‌های کنکور
    """
    
    def __init__(self, data_manager, output_dir='models/', random_state=42):
        """
        Initialize trainer with exam data manager
        
        Parameters:
        -----------
        data_manager : ExamDataManager
            Manager containing prepared exam data
        output_dir : str
            Directory to save models
        random_state : int
            Random seed for reproducibility
        """
        self.data_manager = data_manager
        self.output_dir = output_dir
        self.random_state = random_state
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 دستگاه آموزش: {self.device}")
        
        # Store best models
        self.best_models = {}
    
    def define_model_hyperparameters(self, model_type):
        """
        Define hyperparameter search space for each model type
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'MLP', 'TabTransformer', 'RandomForest', 
                          'XGBoost', 'LightGBM', 'SVM', 'Logistic'
        
        Returns:
        --------
        dict: Hyperparameter space
        """
        print(f"📊 تعریف فضای هایپرپارامتر برای {model_type}")
        
        if model_type == 'MLP':
            return {
                'hidden_layers': [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [1e-3, 1e-4, 5e-4],
                'batch_size': [32, 64, 128],
                'weight_decay': [1e-4, 1e-5],
                'activation': ['relu', 'leaky_relu']
            }
        
        elif model_type == 'TabTransformer':
            return {
                'embedding_dim': [16, 32, 64],
                'num_heads': [4, 8],
                'num_layers': [2, 4, 6],
                'transformer_dropout': [0.1, 0.2],
                'mlp_hidden': [(128, 64), (256, 128)],
                'learning_rate': [1e-3, 1e-4],
                'batch_size': [32, 64],
                'weight_decay': [1e-5]
            }
        
        elif model_type == 'RandomForest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
        
        elif model_type == 'XGBoost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
        
        elif model_type == 'LightGBM':
            return {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 63, 127],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        
        elif model_type == 'SVM':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1],
                'degree': [2, 3, 4],  # for poly kernel
                'probability': [True]
            }
        
        elif model_type == 'Logistic':
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [1000, 2000],
                'l1_ratio': [0.1, 0.5, 0.9]  # for elasticnet
            }
        
        else:
            raise ValueError(f"نوع مدل نامعتبر: {model_type}")
    
    def prepare_data_for_model(self, model_type, train_idx, val_idx):
        """
        Prepare data for specific model type
        
        Parameters:
        -----------
        model_type : str
            Type of model
        train_idx : array
            Training indices
        val_idx : array
            Validation indices
        
        Returns:
        --------
        tuple: Prepared data (X_train, y_train, X_val, y_val)
        """
        print(f"  📦 آماده‌سازی داده برای {model_type}")
        
        if model_type in ['MLP', 'TabTransformer']:
            # For neural networks
            if model_type == 'MLP':
                X_train = self.data_manager.X[train_idx]
                X_val = self.data_manager.X[val_idx]
                return X_train, X_val
            else:  # TabTransformer
                X_train_cat = self.data_manager.X_cat[train_idx]
                X_train_cont = self.data_manager.X_cont[train_idx]
                X_val_cat = self.data_manager.X_cat[val_idx]
                X_val_cont = self.data_manager.X_cont[val_idx]
                
                return (X_train_cat, X_train_cont), (X_val_cat, X_val_cont)
        
        else:
            # For traditional models
            X_train = self.data_manager.X[train_idx]
            X_val = self.data_manager.X[val_idx]
            return X_train, X_val
    
    def create_pytorch_datasets(self, X_train, X_val, y_train, y_val, model_type, batch_size):
        """
        Create PyTorch datasets and dataloaders
        
        Parameters:
        -----------
        X_train : array or tuple
            Training features
        X_val : array or tuple
            Validation features
        y_train : array
            Training labels
        y_val : array
            Validation labels
        model_type : str
            Type of model
        batch_size : int
            Batch size
        
        Returns:
        --------
        tuple: (train_loader, val_loader)
        """
        if model_type == 'MLP':
            train_dataset = ExamTorchDataset(X_train, y_train)
            val_dataset = ExamTorchDataset(X_val, y_val)
        else:  # TabTransformer
            X_cat_train, X_cont_train = X_train
            X_cat_val, X_cont_val = X_val
            train_dataset = ExamTabTransformerDataset(X_cat_train, X_cont_train, y_train)
            val_dataset = ExamTabTransformerDataset(X_cat_val, X_cont_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate evaluation metrics
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_pred_proba : array, optional
            Predicted probabilities
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        metrics = {}
        
        if self.data_manager.task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                except:
                    metrics['roc_auc'] = 0.5
        
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            eps = 1e-10
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100
        
        return metrics
    
    def train_sklearn_model(self, model_type, config, X_train, y_train, X_val, y_val):
        """
        Train scikit-learn model
        
        Parameters:
        -----------
        model_type : str
            Type of model
        config : dict
            Model configuration
        X_train : array
            Training features
        y_train : array
            Training labels
        X_val : array
            Validation features
        y_val : array
            Validation labels
        
        Returns:
        --------
        tuple: (trained_model, val_metrics, val_score)
        """
        print(f"  🤖 آموزش مدل {model_type}")
        
        # Import based on model type
        if model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if self.data_manager.task_type == 'classification':
                model = RandomForestClassifier(**config, random_state=self.random_state)
            else:
                model = RandomForestRegressor(**config, random_state=self.random_state)
        
        elif model_type == 'XGBoost':
            import xgboost as xgb
            if self.data_manager.task_type == 'classification':
                model = xgb.XGBClassifier(**config, random_state=self.random_state, use_label_encoder=False)
            else:
                model = xgb.XGBRegressor(**config, random_state=self.random_state)
        
        elif model_type == 'LightGBM':
            import lightgbm as lgb
            if self.data_manager.task_type == 'classification':
                model = lgb.LGBMClassifier(**config, random_state=self.random_state)
            else:
                model = lgb.LGBMRegressor(**config, random_state=self.random_state)
        
        elif model_type == 'SVM':
            from sklearn.svm import SVC, SVR
            if self.data_manager.task_type == 'classification':
                model = SVC(**config, random_state=self.random_state)
            else:
                model = SVR(**config)
        
        elif model_type == 'Logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**config, random_state=self.random_state)
        
        else:
            raise ValueError(f"نوع مدل نامعتبر: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Get probabilities if available
        y_proba_val = None
        if self.data_manager.task_type == 'classification' and hasattr(model, 'predict_proba'):
            y_proba_val = model.predict_proba(X_val)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        val_metrics = self.calculate_metrics(y_val, y_pred_val, y_proba_val)
        
        # Get validation score for optimization
        if self.data_manager.task_type == 'classification':
            val_score = val_metrics.get('roc_auc', val_metrics.get('roc_auc_ovr', val_metrics['accuracy']))
        else:
            val_score = -val_metrics['rmse']  # Negative for maximization
        
        return model, val_metrics, val_score
    
    def train_pytorch_model(self, model_type, config, train_loader, val_loader):
        """
        Train PyTorch model
        
        Parameters:
        -----------
        model_type : str
            Type of model
        config : dict
            Model configuration
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        
        Returns:
        --------
        tuple: (trained_model, best_val_score, history)
        """
        print(f"  🧠 آموزش مدل {model_type}")
        
        # Initialize model
        if model_type == 'MLP':
            input_dim = self.data_manager.X.shape[1]
            num_classes = self.data_manager.num_classes if self.data_manager.task_type == 'classification' else 1
            
            model = ExamMLP(
                input_dim=input_dim,
                hidden_dims=config['hidden_layers'],
                num_classes=num_classes,
                dropout_rate=config['dropout_rate'],
                activation=config['activation']
            )
        
        elif model_type == 'TabTransformer':
            num_categorical = self.data_manager.X_cat.shape[1]
            num_continuous = self.data_manager.X_cont.shape[1]
            categories = self.data_manager.categories
            num_classes = self.data_manager.num_classes if self.data_manager.task_type == 'classification' else 1
            
            model = ExamTabTransformer(
                num_categorical=num_categorical,
                num_continuous=num_continuous,
                categories=categories,
                num_classes=num_classes,
                embedding_dim=config['embedding_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                transformer_dropout=config['transformer_dropout'],
                mlp_hidden=config['mlp_hidden'],
                mlp_dropout=0.2
            )
        
        elif model_type == 'Regressor':
            input_dim = self.data_manager.X.shape[1]
            model = ExamRegressor(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_layers', (128, 64)),
                dropout_rate=config.get('dropout_rate', 0.2)
            )
        
        else:
            raise ValueError(f"نوع مدل نامعتبر: {model_type}")
        
        model.to(self.device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        if self.data_manager.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=15, verbose=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'epochs': []
        }
        
        best_val_score = -np.inf if self.data_manager.task_type == 'classification' else np.inf
        best_model_state = None
        
        # Training loop
        num_epochs = config.get('max_epochs', 100)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch in train_loader:
                if model_type == 'MLP' or model_type == 'Regressor':
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                
                else:  # TabTransformer
                    X_cat_batch, X_cont_batch, y_batch = batch
                    X_cat_batch = X_cat_batch.to(self.device)
                    X_cont_batch = X_cont_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_cat_batch, X_cont_batch)
                
                if self.data_manager.task_type == 'classification':
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs.squeeze(), y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Store predictions
                if self.data_manager.task_type == 'classification':
                    probs = torch.softmax(outputs, dim=1)
                    train_preds.append(probs.detach().cpu().numpy())
                else:
                    train_preds.append(outputs.detach().cpu().numpy())
                
                train_targets.append(y_batch.detach().cpu().numpy())
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'MLP' or model_type == 'Regressor':
                        X_batch, y_batch = batch
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = model(X_batch)
                    
                    else:  # TabTransformer
                        X_cat_batch, X_cont_batch, y_batch = batch
                        X_cat_batch = X_cat_batch.to(self.device)
                        X_cont_batch = X_cont_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = model(X_cat_batch, X_cont_batch)
                    
                    if self.data_manager.task_type == 'classification':
                        loss = criterion(outputs, y_batch)
                        probs = torch.softmax(outputs, dim=1)
                        val_preds.append(probs.cpu().numpy())
                    else:
                        loss = criterion(outputs.squeeze(), y_batch)
                        val_preds.append(outputs.cpu().numpy())
                    
                    val_loss += loss.item()
                    val_targets.append(y_batch.cpu().numpy())
            
            # Calculate metrics
            train_preds = np.concatenate(train_preds)
            train_targets = np.concatenate(train_targets)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            
            if self.data_manager.task_type == 'classification':
                train_metrics = self.calculate_metrics(train_targets, np.argmax(train_preds, axis=1), train_preds)
                val_metrics = self.calculate_metrics(val_targets, np.argmax(val_preds, axis=1), val_preds)
                val_score = val_metrics.get('roc_auc', val_metrics.get('roc_auc_ovr', val_metrics['accuracy']))
            else:
                train_metrics = self.calculate_metrics(train_targets, train_preds)
                val_metrics = self.calculate_metrics(val_targets, val_preds)
                val_score = -val_metrics['rmse']  # Negative for maximization
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            history['epochs'].append(epoch + 1)
            
            # Check for best model
            if self.data_manager.task_type == 'classification':
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = model.state_dict().copy()
            else:
                if val_score > best_val_score:  # Note: val_score is negative RMSE
                    best_val_score = val_score
                    best_model_state = model.state_dict().copy()
            
            # Early stopping
            early_stopping(val_score, model)
            
            if (epoch + 1) % 10 == 0:
                if self.data_manager.task_type == 'classification':
                    print(f"    📈 Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                          f"Val Loss={val_loss/len(val_loader):.4f}, Val ROC-AUC={val_score:.4f}")
                else:
                    print(f"    📈 Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                          f"Val Loss={val_loss/len(val_loader):.4f}, Val RMSE={-val_score:.4f}")
            
            if early_stopping.early_stop:
                print(f"    🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_val_score, history
    
    def inner_cross_validation(self, model_type, hyperparams, train_idx, k_inner=3):
        """
        Perform inner k-fold cross validation for hyperparameter tuning
        
        Parameters:
        -----------
        model_type : str
            Type of model
        hyperparams : list
            List of hyperparameter combinations
        train_idx : array
            Training indices
        k_inner : int
            Number of inner folds
        
        Returns:
        --------
        tuple: (best_config, best_score, cv_results)
        """
        print(f"  🔍 انجام Inner {k_inner}-fold CV برای {model_type}")
        
        # Initialize k-fold
        if self.data_manager.task_type == 'classification':
            kf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=self.random_state)
            splits = kf.split(self.data_manager.X[train_idx], self.data_manager.y[train_idx])
        else:
            kf = KFold(n_splits=k_inner, shuffle=True, random_state=self.random_state)
            splits = kf.split(self.data_manager.X[train_idx])
        
        best_config = None
        best_score = -np.inf
        cv_results = []
        
        # Grid search over hyperparameters
        for i, params in enumerate(hyperparams):
            print(f"    ⚙️ ترکیب هایپرپارامتر {i+1}/{len(hyperparams)}")
            
            fold_scores = []
            config = self._create_config_dict(model_type, params)
            
            # Inner cross-validation
            for fold, (inner_train_idx, inner_val_idx) in enumerate(splits):
                print(f"      Fold {fold+1}", end='\r')
                
                # Adjust indices
                actual_inner_train = train_idx[inner_train_idx]
                actual_inner_val = train_idx[inner_val_idx]
                
                # Get data
                y_train_inner = self.data_manager.y[actual_inner_train]
                y_val_inner = self.data_manager.y[actual_inner_val]
                
                # Prepare data
                if model_type in ['MLP', 'TabTransformer', 'Regressor']:
                    X_train_data, X_val_data = self.prepare_data_for_model(
                        model_type, actual_inner_train, actual_inner_val
                    )
                    
                    # Create dataloaders
                    batch_size = config.get('batch_size', 32)
                    train_loader, val_loader = self.create_pytorch_datasets(
                        X_train_data, X_val_data, y_train_inner, y_val_inner,
                        model_type, batch_size
                    )
                    
                    # Train PyTorch model
                    _, val_score, _ = self.train_pytorch_model(
                        model_type, config, train_loader, val_loader
                    )
                
                else:
                    # Traditional models
                    X_train_data, X_val_data = self.prepare_data_for_model(
                        model_type, actual_inner_train, actual_inner_val
                    )
                    
                    # Train sklearn model
                    _, _, val_score = self.train_sklearn_model(
                        model_type, config, X_train_data, y_train_inner,
                        X_val_data, y_val_inner
                    )
                
                fold_scores.append(val_score)
            
            # Calculate mean score for this hyperparameter combination
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            cv_results.append({
                'config': config,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
            
            print(f"    ✅ ترکیب {i+1}: امتیاز={mean_score:.4f} ± {std_score:.4f}")
            
            # Update best configuration
            if mean_score > best_score:
                best_score = mean_score
                best_config = config
        
        print(f"  🏆 بهترین ترکیب: امتیاز={best_score:.4f}")
        
        return best_config, best_score, cv_results
    
    def _create_config_dict(self, model_type, params):
        """Create configuration dictionary from parameter tuple"""
        if model_type == 'MLP':
            keys = ['hidden_layers', 'dropout_rate', 'learning_rate', 
                   'batch_size', 'weight_decay', 'activation']
        elif model_type == 'TabTransformer':
            keys = ['embedding_dim', 'num_heads', 'num_layers', 'transformer_dropout',
                   'mlp_hidden', 'learning_rate', 'batch_size', 'weight_decay']
        elif model_type == 'Regressor':
            keys = ['hidden_layers', 'dropout_rate', 'learning_rate', 
                   'batch_size', 'weight_decay']
        elif model_type == 'RandomForest':
            keys = ['n_estimators', 'max_depth', 'min_samples_split',
                   'min_samples_leaf', 'max_features', 'bootstrap']
        elif model_type == 'XGBoost':
            keys = ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                   'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']
        elif model_type == 'LightGBM':
            keys = ['n_estimators', 'num_leaves', 'max_depth', 'learning_rate',
                   'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
        elif model_type == 'SVM':
            keys = ['C', 'kernel', 'gamma', 'degree', 'probability']
        elif model_type == 'Logistic':
            keys = ['C', 'penalty', 'solver', 'max_iter', 'l1_ratio']
        else:
            raise ValueError(f"نوع مدل نامعتبر: {model_type}")
        
        return dict(zip(keys, params))
    
    def nested_cross_validation(self, model_type, k_outer=5, k_inner=3):
        """
        Perform nested cross-validation for model evaluation
        
        Parameters:
        -----------
        model_type : str
            Type of model
        k_outer : int
            Number of outer folds
        k_inner : int
            Number of inner folds
        
        Returns:
        --------
        dict: Results containing scores and best models
        """
        print(f"\n🎯 شروع Nested Cross-Validation برای {model_type}")
        print("="*60)
        
        # Define hyperparameter space
        param_space = self.define_model_hyperparameters(model_type)
        
        # Generate all combinations
        param_combinations = list(itertools.product(*param_space.values()))
        print(f"  تعداد ترکیب‌های هایپرپارامتر: {len(param_combinations)}")
        
        # Initialize stratified k-fold for outer loop
        if self.data_manager.task_type == 'classification':
            outer_kf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=self.random_state)
            outer_splits = outer_kf.split(self.data_manager.X, self.data_manager.y)
        else:
            outer_kf = KFold(n_splits=k_outer, shuffle=True, random_state=self.random_state)
            outer_splits = outer_kf.split(self.data_manager.X)
        
        # Store results
        results = {
            'model_type': model_type,
            'outer_folds': [],
            'best_configs': [],
            'train_scores': [],
            'val_scores': [],
            'test_scores': [],
            'test_metrics': [],
            'best_models': []
        }
        
        # Outer cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(outer_splits):
            print(f"\n📁 Outer Fold {fold + 1}/{k_outer}")
            print("-"*40)
            
            # Inner CV for hyperparameter tuning
            print("  🔧 تنظیم هایپرپارامتر با Inner CV...")
            best_config, best_inner_score, cv_results = self.inner_cross_validation(
                model_type, param_combinations, train_idx, k_inner
            )
            
            # Train final model with best configuration on entire training set
            print("  🚀 آموزش مدل نهایی با بهترین هایپرپارامتر...")
            
            # Get data
            y_train = self.data_manager.y[train_idx]
            y_test = self.data_manager.y[test_idx]
            
            if model_type in ['MLP', 'TabTransformer', 'Regressor']:
                # PyTorch model
                X_train_data, X_test_data = self.prepare_data_for_model(
                    model_type, train_idx, test_idx
                )
                
                # Create dataloaders
                batch_size = best_config.get('batch_size', 32)
                train_loader, test_loader = self.create_pytorch_datasets(
                    X_train_data, X_test_data, y_train, y_test,
                    model_type, batch_size
                )
                
                # Train final model
                final_model, final_val_score, history = self.train_pytorch_model(
                    model_type, best_config, train_loader, test_loader
                )
                
                # Final evaluation on test set
                final_model.eval()
                test_preds = []
                test_targets = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        if model_type == 'MLP' or model_type == 'Regressor':
                            X_batch, y_batch = batch
                            X_batch = X_batch.to(self.device)
                            outputs = final_model(X_batch)
                        else:  # TabTransformer
                            X_cat_batch, X_cont_batch, y_batch = batch
                            X_cat_batch = X_cat_batch.to(self.device)
                            X_cont_batch = X_cont_batch.to(self.device)
                            outputs = final_model(X_cat_batch, X_cont_batch)
                        
                        if self.data_manager.task_type == 'classification':
                            probs = torch.softmax(outputs, dim=1)
                            test_preds.append(probs.cpu().numpy())
                        else:
                            test_preds.append(outputs.cpu().numpy())
                        
                        test_targets.append(y_batch.cpu().numpy())
                
                test_preds = np.concatenate(test_preds)
                test_targets = np.concatenate(test_targets)
                
                if self.data_manager.task_type == 'classification':
                    test_metrics = self.calculate_metrics(
                        test_targets, np.argmax(test_preds, axis=1), test_preds
                    )
                    test_score = test_metrics.get('roc_auc', test_metrics.get('roc_auc_ovr', test_metrics['accuracy']))
                else:
                    test_metrics = self.calculate_metrics(test_targets, test_preds)
                    test_score = test_metrics['rmse']
            
            else:
                # Traditional model
                X_train_data, X_test_data = self.prepare_data_for_model(
                    model_type, train_idx, test_idx
                )
                
                # Train final model
                final_model, _, _ = self.train_sklearn_model(
                    model_type, best_config, X_train_data, y_train,
                    X_test_data, y_test
                )
                
                # Make predictions on test set
                y_pred_test = final_model.predict(X_test_data)
                
                if self.data_manager.task_type == 'classification' and hasattr(final_model, 'predict_proba'):
                    y_proba_test = final_model.predict_proba(X_test_data)
                    test_metrics = self.calculate_metrics(y_test, y_pred_test, y_proba_test)
                    test_score = test_metrics.get('roc_auc', test_metrics.get('roc_auc_ovr', test_metrics['accuracy']))
                else:
                    test_metrics = self.calculate_metrics(y_test, y_pred_test)
                    if self.data_manager.task_type == 'classification':
                        test_score = test_metrics['accuracy']
                    else:
                        test_score = test_metrics['rmse']
            
            # Store results for this fold
            results['outer_folds'].append(fold)
            results['best_configs'].append(best_config)
            results['train_scores'].append(best_inner_score)
            results['val_scores'].append(best_inner_score)
            results['test_scores'].append(test_score)
            results['test_metrics'].append(test_metrics)
            results['best_models'].append(final_model)
            
            print(f"\n  📊 نتایج Fold {fold + 1}:")
            print(f"    🔍 امتیاز اعتبارسنجی: {best_inner_score:.4f}")
            print(f"    🧪 امتیاز تست: {test_score:.4f}")
            
            # Save model for this fold
            model_path = os.path.join(self.output_dir, f"{model_type}_fold{fold+1}.pt")
            if model_type in ['MLP', 'TabTransformer', 'Regressor']:
                torch.save({
                    'model_state_dict': final_model.state_dict(),
                    'config': best_config,
                    'history': history
                }, model_path)
            else:
                import joblib
                joblib.dump(final_model, model_path.replace('.pt', '.joblib'))
            
            print(f"    💾 مدل در {model_path} ذخیره شد")
        
        # Calculate final statistics
        results['mean_train_score'] = np.mean(results['train_scores'])
        results['std_train_score'] = np.std(results['train_scores'])
        results['mean_val_score'] = np.mean(results['val_scores'])
        results['std_val_score'] = np.std(results['val_scores'])
        results['mean_test_score'] = np.mean(results['test_scores'])
        results['std_test_score'] = np.std(results['test_scores'])
        
        # Find best overall model
        if self.data_manager.task_type == 'classification':
            best_fold_idx = np.argmax(results['test_scores'])
        else:
            best_fold_idx = np.argmin(results['test_scores'])
        
        results['best_overall_model'] = results['best_models'][best_fold_idx]
        results['best_overall_config'] = results['best_configs'][best_fold_idx]
        results['best_overall_score'] = results['test_scores'][best_fold_idx]
        results['best_fold'] = best_fold_idx + 1
        
        print(f"\n✅ Nested CV برای {model_type} کامل شد")
        print("="*60)
        print(f"📈 میانگین امتیاز آموزش: {results['mean_train_score']:.4f} ± {results['std_train_score']:.4f}")
        print(f"📊 میانگین امتیاز اعتبارسنجی: {results['mean_val_score']:.4f} ± {results['std_val_score']:.4f}")
        print(f"🧪 میانگین امتیاز تست: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
        print(f"🏆 بهترین Fold: {results['best_fold']} با امتیاز: {results['best_overall_score']:.4f}")
        
        # Store results
        self.results[model_type] = results
        self.best_models[model_type] = results['best_overall_model']
        
        return results
    
    def compare_models(self, model_types=None):
        """
        Compare multiple models and return summary
        
        Parameters:
        -----------
        model_types : list, optional
            List of model types to compare
        
        Returns:
        --------
        pd.DataFrame: Comparison results
        """
        if model_types is None:
            model_types = ['RandomForest', 'XGBoost', 'LightGBM', 'MLP', 'TabTransformer']
        
        comparison = []
        
        for model_type in model_types:
            if model_type in self.results:
                results = self.results[model_type]
                
                if self.data_manager.task_type == 'classification':
                    comparison.append({
                        'Model': model_type,
                        'Test AUC': f"{results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}",
                        'Best AUC': f"{results['best_overall_score']:.4f}",
                        'Best Fold': results['best_fold']
                    })
                else:
                    comparison.append({
                        'Model': model_type,
                        'Test RMSE': f"{results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}",
                        'Best RMSE': f"{results['best_overall_score']:.4f}",
                        'Best Fold': results['best_fold']
                    })
        
        df_comparison = pd.DataFrame(comparison)
        
        # Plot comparison
        self._plot_model_comparison(df_comparison)
        
        return df_comparison
    
    def _plot_model_comparison(self, comparison_df):
        """Plot model comparison"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        models = comparison_df['Model']
        
        if self.data_manager.task_type == 'classification':
            scores = [float(x.split(' ± ')[0]) for x in comparison_df['Test AUC']]
            stds = [float(x.split(' ± ')[1]) for x in comparison_df['Test AUC']]
            title = 'مقایسه AUC مدل‌ها'
            ylabel = 'AUC Score'
        else:
            scores = [float(x.split(' ± ')[0]) for x in comparison_df['Test RMSE']]
            stds = [float(x.split(' ± ')[1]) for x in comparison_df['Test RMSE']]
            title = 'مقایسه RMSE مدل‌ها (کمتر بهتر است)'
            ylabel = 'RMSE'
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = plt.bar(models, scores, yerr=stds, capsize=10, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'model_comparison.jpg')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 نمودار مقایسه در {plot_path} ذخیره شد")
    
    def predict_with_best_model(self, model_type, X_new):
        """
        Make predictions with the best model of given type
        
        Parameters:
        -----------
        model_type : str
            Type of model
        X_new : array
            New data to predict
        
        Returns:
        --------
        array: Predictions
        """
        if model_type not in self.best_models:
            raise ValueError(f"مدل {model_type} یافت نشد. ابتدا مدل را آموزش دهید.")
        
        model = self.best_models[model_type]
        
        if model_type in ['MLP', 'TabTransformer', 'Regressor']:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_new).to(self.device)
                
                if model_type == 'TabTransformer':
                    # For TabTransformer, we need to separate categorical and continuous
                    # This requires additional handling - assuming X_new is already prepared
                    # In practice, you'd need to use the same preprocessing
                    num_cat = self.data_manager.X_cat.shape[1]
                    X_cat = X_tensor[:, :num_cat].long()
                    X_cont = X_tensor[:, num_cat:]
                    outputs = model(X_cat, X_cont)
                else:
                    outputs = model(X_tensor)
                
                if self.data_manager.task_type == 'classification':
                    probs = torch.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
                else:
                    return outputs.cpu().numpy()
        else:
            return model.predict(X_new)