"""
ارزیابی مدل‌های یادگیری ماشین برای داده‌های کنکور ایران
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

from exam_dataset import ExamTorchDataset, ExamTabTransformerDataset


class ExamModelEvaluator:
    """
    کلاس جامع برای ارزیابی مدل‌های کنکور
    """
    
    def __init__(self, data_manager, output_dir='evaluation_results/'):
        """
        Initialize evaluator with exam data manager
        
        Parameters:
        -----------
        data_manager : ExamDataManager
            Manager containing prepared exam data
        output_dir : str
            Directory to save evaluation results
        """
        self.data_manager = data_manager
        self.output_dir = output_dir
        self.task_type = data_manager.task_type
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # Set device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store evaluation results
        self.results = {}
        
        print(f"📊 ارزیاب مدل‌ها ایجاد شد: output_dir={output_dir}")
    
    def load_model(self, model_path, model_type):
        """
        Load trained model from file
        
        Parameters:
        -----------
        model_path : str
            Path to saved model
        model_type : str
            Type of model: 'MLP', 'TabTransformer', 'sklearn'
        
        Returns:
        --------
        object: Loaded model
        """
        print(f"📂 در حال بارگذاری مدل از {model_path}")
        
        if model_type in ['MLP', 'TabTransformer', 'Regressor']:
            # Load PyTorch model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model architecture
            if model_type == 'MLP':
                from exam_models import ExamMLP
                model = ExamMLP(**checkpoint['config'])
            elif model_type == 'TabTransformer':
                from exam_models import ExamTabTransformer
                model = ExamTabTransformer(**checkpoint['config'])
            else:  # Regressor
                from exam_models import ExamRegressor
                model = ExamRegressor(**checkpoint['config'])
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Also return config and history if needed
            return {
                'model': model,
                'config': checkpoint['config'],
                'history': checkpoint.get('history', {}),
                'type': model_type
            }
        
        else:
            # Load scikit-learn model
            import joblib
            model = joblib.load(model_path)
            return {
                'model': model,
                'type': model_type,
                'config': {}
            }
    
    def load_scaler(self, scaler_path):
        """
        Load scaler used during training
        
        Parameters:
        -----------
        scaler_path : str
            Path to saved scaler
        
        Returns:
        --------
        object: Loaded scaler
        """
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler بارگذاری شد")
            return scaler
        else:
            print(f"⚠️ Scaler یافت نشد: {scaler_path}")
            return None
    
    def prepare_test_data(self, model_info, test_idx, scaler=None):
        """
        Prepare test data for evaluation
        
        Parameters:
        -----------
        model_info : dict
            Model information
        test_idx : array
            Test indices
        scaler : object, optional
            Scaler to transform data
        
        Returns:
        --------
        tuple: Prepared test data
        """
        model_type = model_info['type']
        print(f"📊 آماده‌سازی داده تست برای {model_type}")
        
        if model_type == 'MLP' or model_type == 'Regressor':
            # For MLP/Regressor: continuous features only
            X_test = self.data_manager.X[test_idx]
            if scaler:
                X_test = scaler.transform(X_test)
            y_test = self.data_manager.y[test_idx]
            
            return X_test, y_test, None
        
        elif model_type == 'TabTransformer':
            # For TabTransformer: categorical + continuous features
            X_cat_test = self.data_manager.X_cat[test_idx]
            X_cont_test = self.data_manager.X_cont[test_idx]
            
            if scaler:
                X_cont_test = scaler.transform(X_cont_test)
            
            y_test = self.data_manager.y[test_idx]
            
            return (X_cat_test, X_cont_test), y_test, None
        
        else:
            # For scikit-learn models
            X_test = self.data_manager.X[test_idx]
            if scaler:
                X_test = scaler.transform(X_test)
            y_test = self.data_manager.y[test_idx]
            
            return X_test, y_test, None
    
    def evaluate_classification_model(self, model_info, X_test, y_test, model_name, fold=None):
        """
        Evaluate classification model
        
        Parameters:
        -----------
        model_info : dict
            Model information
        X_test : array or tuple
            Test features
        y_test : array
            Test labels
        model_name : str
            Name of the model
        fold : int, optional
            Fold number
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        print(f"🧪 ارزیابی مدل طبقه‌بندی: {model_name}")
        
        model_type = model_info['type']
        model = model_info['model']
        
        # Make predictions
        if model_type in ['MLP', 'TabTransformer', 'Regressor']:
            # PyTorch models
            if model_type == 'MLP' or model_type == 'Regressor':
                test_dataset = ExamTorchDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                y_pred_proba, y_pred, y_true = self._predict_pytorch_mlp(model, test_loader)
            
            else:  # TabTransformer
                X_cat_test, X_cont_test = X_test
                test_dataset = ExamTabTransformerDataset(X_cat_test, X_cont_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                y_pred_proba, y_pred, y_true = self._predict_pytorch_tabtransformer(model, test_loader)
        
        else:
            # scikit-learn models
            y_true = y_test
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
                # For models without predict_proba, create dummy probabilities
                y_pred_proba = np.zeros((len(y_pred), len(np.unique(y_true))))
                for i, pred in enumerate(y_pred):
                    y_pred_proba[i, pred] = 1.0
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate classification report
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot results
        plot_dir = os.path.join(self.output_dir, 'plots')
        self._plot_classification_results(
            y_true, y_pred_proba, cm, model_name, fold, metrics, plot_dir
        )
        
        # Store results
        result_key = f"{model_name}_fold{fold}" if fold is not None else model_name
        self.results[result_key] = {
            'metrics': metrics,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'true': y_true.tolist(),
                'pred': y_pred.tolist(),
                'pred_proba': y_pred_proba.tolist()
            },
            'model_type': model_type,
            'model_name': model_name
        }
        
        print(f"✅ ارزیابی کامل شد")
        print(f"   دقت: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc_ovr']:.4f}")
        
        return metrics
    
    def _predict_pytorch_mlp(self, model, data_loader):
        """
        Make predictions with PyTorch MLP model
        """
        model.eval()
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                
                outputs = model(data)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        return all_probs, all_preds, all_targets
    
    def _predict_pytorch_tabtransformer(self, model, data_loader):
        """
        Make predictions with PyTorch TabTransformer model
        """
        model.eval()
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for cat_data, cont_data, targets in data_loader:
                cat_data = cat_data.to(self.device)
                cont_data = cont_data.to(self.device)
                
                outputs = model(cat_data, cont_data)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        return all_probs, all_preds, all_targets
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['roc_auc_ovr'] = metrics['roc_auc']
            else:
                # Multiclass classification
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
        except:
            metrics['roc_auc_ovr'] = 0.5
            metrics['roc_auc_ovo'] = 0.5
        
        return metrics
    
    def evaluate_regression_model(self, model_info, X_test, y_test, model_name, fold=None):
        """
        Evaluate regression model
        
        Parameters:
        -----------
        model_info : dict
            Model information
        X_test : array or tuple
            Test features
        y_test : array
            Test labels
        model_name : str
            Name of the model
        fold : int, optional
            Fold number
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        print(f"🧪 ارزیابی مدل رگرسیون: {model_name}")
        
        model_type = model_info['type']
        model = model_info['model']
        
        # Make predictions
        if model_type in ['MLP', 'TabTransformer', 'Regressor']:
            # PyTorch models
            if model_type == 'MLP' or model_type == 'Regressor':
                test_dataset = ExamTorchDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                y_pred, y_true = self._predict_pytorch_regression(model, test_loader)
            
            else:  # TabTransformer
                X_cat_test, X_cont_test = X_test
                test_dataset = ExamTabTransformerDataset(X_cat_test, X_cont_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                y_pred, y_true = self._predict_pytorch_regression_tabtransformer(model, test_loader)
        
        else:
            # scikit-learn models
            y_true = y_test
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_true, y_pred)
        
        # Plot results
        plot_dir = os.path.join(self.output_dir, 'plots')
        self._plot_regression_results(y_true, y_pred, model_name, fold, metrics, plot_dir)
        
        # Store results
        result_key = f"{model_name}_fold{fold}" if fold is not None else model_name
        self.results[result_key] = {
            'metrics': metrics,
            'predictions': {
                'true': y_true.tolist(),
                'pred': y_pred.tolist()
            },
            'model_type': model_type,
            'model_name': model_name
        }
        
        print(f"✅ ارزیابی کامل شد")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        print(f"   R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def _predict_pytorch_regression(self, model, data_loader):
        """Make regression predictions with MLP/Regressor"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                outputs = model(data)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets)
        
        return all_preds, all_targets
    
    def _predict_pytorch_regression_tabtransformer(self, model, data_loader):
        """Make regression predictions with TabTransformer"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for cat_data, cont_data, targets in data_loader:
                cat_data = cat_data.to(self.device)
                cont_data = cont_data.to(self.device)
                outputs = model(cat_data, cont_data)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets)
        
        return all_preds, all_targets
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
        metrics['mape'] = mape
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = correlation
        
        return metrics
    
    def _plot_classification_results(self, y_true, y_pred_proba, cm, model_name, fold, metrics, save_dir):
        """
        Plot classification results
        """
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Class {i}' for i in range(cm.shape[0])],
                   yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_path = os.path.join(save_dir, f'{model_name}_fold{fold}_confusion_matrix.jpg' if fold else f'{model_name}_confusion_matrix.jpg')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        if len(np.unique(y_true)) <= 10:
            plt.figure(figsize=(10, 8))
            
            if len(np.unique(y_true)) == 2:
                # Binary ROC
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = metrics['roc_auc']
                
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            else:
                # Multiclass ROC
                from sklearn.preprocessing import label_binarize
                
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                
                fpr = {}
                tpr = {}
                roc_auc = {}
                
                colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
                
                for i, cls in enumerate(classes):
                    fpr[cls], tpr[cls], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc[cls] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    
                    plt.plot(fpr[cls], tpr[cls], color=colors[i], lw=2,
                            label=f'Class {cls} (AUC = {roc_auc[cls]:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            roc_path = os.path.join(save_dir, f'{model_name}_fold{fold}_roc_curve.jpg' if fold else f'{model_name}_roc_curve.jpg')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Metrics Bar Plot
        plt.figure(figsize=(12, 6))
        
        metric_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
        metric_values = [metrics['accuracy'], metrics['f1_macro'], metrics['roc_auc_ovr']]
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        bars = plt.bar(metric_names, metric_values, color=colors, edgecolor='black')
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(f'Evaluation Metrics - {model_name}')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        metrics_path = os.path.join(save_dir, f'{model_name}_fold{fold}_metrics.jpg' if fold else f'{model_name}_metrics.jpg')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regression_results(self, y_true, y_pred, model_name, fold, metrics, save_dir):
        """
        Plot regression results
        """
        # 1. True vs Predicted Scatter Plot
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"RMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nR²: {metrics['r2']:.3f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        scatter_path = os.path.join(save_dir, f'{model_name}_fold{fold}_scatter.jpg' if fold else f'{model_name}_scatter.jpg')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residual Plot
        plt.figure(figsize=(10, 8))
        
        residuals = y_true - y_pred
        
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {model_name}')
        plt.grid(True, alpha=0.3)
        
        residual_path = os.path.join(save_dir, f'{model_name}_fold{fold}_residuals.jpg' if fold else f'{model_name}_residuals.jpg')
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model_fold(self, model_path, model_type, fold, test_idx, scaler_path=None):
        """
        Evaluate model for a specific fold
        
        Parameters:
        -----------
        model_path : str
            Path to saved model
        model_type : str
            Type of model
        fold : int
            Fold number
        test_idx : array
            Test indices
        scaler_path : str, optional
            Path to saved scaler
        
        Returns:
        --------
        dict: Evaluation results
        """
        print(f"\n📊 ارزیابی مدل Fold {fold}")
        print("-"*40)
        
        # Load model
        model_info = self.load_model(model_path, model_type)
        
        # Load scaler if provided
        scaler = None
        if scaler_path:
            scaler = self.load_scaler(scaler_path)
        
        # Prepare test data
        test_data = self.prepare_test_data(model_info, test_idx, scaler)
        
        # Evaluate based on task type
        model_name = f"{model_type}_fold{fold}"
        
        if self.task_type == 'classification':
            X_test, y_test, _ = test_data
            results = self.evaluate_classification_model(
                model_info, X_test, y_test, model_name, fold
            )
        else:
            X_test, y_test, _ = test_data
            results = self.evaluate_regression_model(
                model_info, X_test, y_test, model_name, fold
            )
        
        return results
    
    def evaluate_all_models(self, models_dir='models/', output_dir='evaluation_results/'):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        output_dir : str
            Output directory for evaluation results
        
        Returns:
        --------
        dict: All evaluation results
        """
        print("\n🧪 شروع ارزیابی همه مدل‌ها")
        print("="*60)
        
        # Find all model files
        import glob
        
        model_files = []
        model_files.extend(glob.glob(os.path.join(models_dir, '*.pt')))
        model_files.extend(glob.glob(os.path.join(models_dir, '*.pkl')))
        model_files.extend(glob.glob(os.path.join(models_dir, '*.joblib')))
        
        # Group by model type and fold
        model_groups = {}
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            
            # Parse filename to get model type and fold
            if 'fold' in filename:
                # Format: ModelType_foldX.pt
                parts = filename.split('_')
                model_type = parts[0]
                
                # Extract fold number
                fold_part = [p for p in parts if 'fold' in p][0]
                fold_num = int(fold_part.replace('fold', '').replace('.pt', '').replace('.pkl', '').replace('.joblib', ''))
            else:
                # Single model file
                model_type = filename.split('.')[0]
                fold_num = 0
            
            if model_type not in model_groups:
                model_groups[model_type] = []
            
            model_groups[model_type].append((fold_num, model_file))
        
        # Evaluate each model
        all_results = {}
        
        for model_type, model_list in model_groups.items():
            print(f"\n📊 ارزیابی مدل‌های {model_type}")
            
            fold_results = []
            
            for fold_num, model_path in sorted(model_list):
                print(f"  🔍 Fold {fold_num}")
                
                # For demonstration, use last 20% as test set
                # In practice, you should use the actual test indices from training
                n_samples = len(self.data_manager.X)
                test_size = int(n_samples * 0.2)
                
                if hasattr(self.data_manager, 'test_indices') and self.data_manager.test_indices is not None:
                    test_idx = self.data_manager.test_indices
                else:
                    np.random.seed(42)
                    test_idx = np.random.choice(n_samples, test_size, replace=False)
                
                try:
                    # Look for scaler file
                    scaler_path = None
                    possible_scaler = model_path.replace('.pt', '_scaler.pkl').replace('.pkl', '_scaler.pkl').replace('.joblib', '_scaler.pkl')
                    if os.path.exists(possible_scaler):
                        scaler_path = possible_scaler
                    
                    result = self.evaluate_model_fold(
                        model_path, model_type, fold_num, test_idx, scaler_path
                    )
                    fold_results.append(result)
                    
                except Exception as e:
                    print(f"    ❌ خطا در ارزیابی: {e}")
                    import traceback
                    traceback.print_exc()
                    fold_results.append(None)
            
            # Store results for this model type
            all_results[model_type] = {
                'fold_results': fold_results,
                'num_folds': len([r for r in fold_results if r is not None])
            }
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        print("\n✅ ارزیابی همه مدل‌ها با موفقیت انجام شد!")
        
        return all_results
    
    def _generate_summary_report(self, all_results):
        """
        Generate summary report of all evaluations
        """
        report_path = os.path.join(self.output_dir, 'reports', 'evaluation_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("📋 گزارش جامع ارزیابی مدل‌های کنکور\n")
            f.write("="*70 + "\n\n")
            f.write(f"تاریخ تولید: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"نوع وظیفه: {self.task_type}\n\n")
            
            summary_data = []
            
            for model_type, results in all_results.items():
                fold_results = [r for r in results['fold_results'] if r is not None]
                
                if not fold_results:
                    continue
                
                f.write(f"\n🎯 مدل: {model_type}\n")
                f.write("-"*40 + "\n")
                f.write(f"تعداد Folds ارزیابی شده: {len(fold_results)}\n\n")
                
                # Collect metrics across folds
                if self.task_type == 'classification':
                    metrics_list = [r['metrics'] for r in fold_results]
                    
                    avg_accuracy = np.mean([m['accuracy'] for m in metrics_list])
                    std_accuracy = np.std([m['accuracy'] for m in metrics_list])
                    
                    avg_f1 = np.mean([m['f1_macro'] for m in metrics_list])
                    std_f1 = np.std([m['f1_macro'] for m in metrics_list])
                    
                    avg_auc = np.mean([m['roc_auc_ovr'] for m in metrics_list])
                    std_auc = np.std([m['roc_auc_ovr'] for m in metrics_list])
                    
                    f.write(f"میانگین Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
                    f.write(f"میانگین F1-Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
                    f.write(f"میانگین ROC-AUC: {avg_auc:.4f} ± {std_auc:.4f}\n")
                    
                    summary_data.append({
                        'Model': model_type,
                        'Accuracy': f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
                        'F1-Score': f"{avg_f1:.4f} ± {std_f1:.4f}",
                        'ROC-AUC': f"{avg_auc:.4f} ± {std_auc:.4f}"
                    })
                
                else:  # regression
                    metrics_list = [r['metrics'] for r in fold_results]
                    
                    avg_rmse = np.mean([m['rmse'] for m in metrics_list])
                    std_rmse = np.std([m['rmse'] for m in metrics_list])
                    
                    avg_mae = np.mean([m['mae'] for m in metrics_list])
                    std_mae = np.std([m['mae'] for m in metrics_list])
                    
                    avg_r2 = np.mean([m['r2'] for m in metrics_list])
                    std_r2 = np.std([m['r2'] for m in metrics_list])
                    
                    f.write(f"میانگین RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}\n")
                    f.write(f"میانگین MAE: {avg_mae:.2f} ± {std_mae:.2f}\n")
                    f.write(f"میانگین R²: {avg_r2:.4f} ± {std_r2:.4f}\n")
                    
                    summary_data.append({
                        'Model': model_type,
                        'RMSE': f"{avg_rmse:.2f} ± {std_rmse:.2f}",
                        'MAE': f"{avg_mae:.2f} ± {std_mae:.2f}",
                        'R²': f"{avg_r2:.4f} ± {std_r2:.4f}"
                    })
            
            # Create summary dataframe
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_csv = os.path.join(self.output_dir, 'reports', 'evaluation_summary.csv')
                summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
                
                f.write("\n" + "="*70 + "\n")
                f.write("📊 خلاصه نتایج:\n")
                f.write(str(summary_df.to_string()) + "\n")
        
        print(f"✅ گزارش ارزیابی در {report_path} ذخیره شد")
    
    def save_evaluation_results(self, filename='evaluation_results.json'):
        """
        Save all evaluation results to JSON file
        """
        file_path = os.path.join(self.output_dir, 'reports', filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ نتایج ارزیابی در {file_path} ذخیره شد")
        
        return file_path
    
    def load_evaluation_results(self, filename='evaluation_results.json'):
        """
        Load evaluation results from JSON file
        """
        file_path = os.path.join(self.output_dir, 'reports', filename)
        
        if not os.path.exists(file_path):
            print(f"❌ فایل {file_path} یافت نشد")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        print(f"✅ نتایج ارزیابی از {file_path} بارگذاری شد")
        
        return self.results