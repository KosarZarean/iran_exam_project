"""
توابع مصورسازی برای داده‌های کنکور ایران
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class ExamVisualizer:
    """
    کلاس مصورسازی برای داده‌های کنکور
    """
    
    def __init__(self, output_dir='visualizations'):
        """
        Parameters:
        -----------
        output_dir : str
            پوشه خروجی برای ذخیره نمودارها
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # تنظیم استایل
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_distribution(self, data, column, title=None, save=True):
        """
        رسم توزیع یک ستون
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(data[column], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'توزیع {column}')
        plt.xlabel(column)
        plt.ylabel('تعداد')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(data[column])
        plt.title(f'Boxplot {column}')
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        
        if title:
            plt.suptitle(title)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'distribution_{column}.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_categorical(self, data, column, top_n=20, title=None, save=True):
        """
        رسم توزیع یک ستون دسته‌ای
        """
        value_counts = data[column].value_counts()
        
        if len(value_counts) > top_n:
            value_counts = value_counts.head(top_n)
            plot_title = f'توزیع {column} ({top_n} دسته برتر)'
        else:
            plot_title = f'توزیع {column}'
        
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        plt.bar(range(len(value_counts)), value_counts.values, color=colors, edgecolor='black')
        
        plt.title(title or plot_title)
        plt.xlabel(column)
        plt.ylabel('تعداد')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'categorical_{column}.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return value_counts
    
    def plot_correlation_heatmap(self, data, columns=None, title=None, save=True):
        """
        رسم heatmap همبستگی
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        if len(columns) < 2:
            print("⚠️ حداقل به 2 ستون عددی نیاز است")
            return
        
        corr_matrix = data[columns].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title(title or 'ماتریس همبستگی ویژگی‌های عددی')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'correlation_heatmap.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return corr_matrix
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title=None, save=True):
        """
        رسم confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title or 'Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, class_names=None, title=None, save=True):
        """
        رسم منحنی ROC
        """
        plt.figure(figsize=(10, 8))
        
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
        else:
            # Multiclass classification
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title or 'ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'roc_curve.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_tsne(self, X, y, title=None, save=True, perplexity=30):
        """
        رسم نمودار t-SNE
        """
        print("🔄 در حال محاسبه t-SNE...")
        
        # اعمال t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                             c=y, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Target')
        
        plt.title(title or 't-SNE Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'tsne_visualization.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return X_tsne
    
    def plot_feature_importance(self, importance_dict, title=None, top_n=20, save=True):
        """
        رسم اهمیت ویژگی‌ها
        """
        # مرتب‌سازی
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_items) > top_n:
            sorted_items = sorted_items[:top_n]
            plot_title = f'{title or "Feature Importance"} (Top {top_n})'
        else:
            plot_title = title or 'Feature Importance'
        
        features = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        plt.figure(figsize=(12, max(6, len(features)*0.3)))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        plt.barh(range(len(features)), scores, color=colors, edgecolor='black')
        
        plt.title(plot_title)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.yticks(range(len(features)), features)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'feature_importance.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_training_history(self, history, title=None, save=True):
        """
        رسم تاریخچه آموزش
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy/Score curves
        if 'train_metrics' in history and history['train_metrics']:
            train_metrics = history['train_metrics']
            val_metrics = history['val_metrics']
            
            # Extract primary metric
            if 'accuracy' in train_metrics[0]:
                train_scores = [m['accuracy'] for m in train_metrics]
                val_scores = [m['accuracy'] for m in val_metrics]
                metric_name = 'Accuracy'
            elif 'roc_auc' in train_metrics[0]:
                train_scores = [m['roc_auc'] for m in train_metrics]
                val_scores = [m['roc_auc'] for m in val_metrics]
                metric_name = 'ROC-AUC'
            elif 'rmse' in train_metrics[0]:
                train_scores = [m['rmse'] for m in train_metrics]
                val_scores = [m['rmse'] for m in val_metrics]
                metric_name = 'RMSE'
            else:
                train_scores = []
                val_scores = []
                metric_name = 'Score'
            
            if train_scores:
                axes[0, 1].plot(epochs, train_scores, 'b-', label=f'Train {metric_name}', linewidth=2)
                axes[0, 1].plot(epochs, val_scores, 'r-', label=f'Val {metric_name}', linewidth=2)
                axes[0, 1].set_title(f'{metric_name} Curves')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel(metric_name)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norm (if available)
        if 'grad_norm' in history:
            axes[1, 1].plot(epochs, history['grad_norm'], 'm-', linewidth=2)
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Training History', fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'training_history.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_model_comparison(self, results_df, metric='accuracy', title=None, save=True):
        """
        رسم مقایسه مدل‌ها
        """
        plt.figure(figsize=(12, 6))
        
        models = results_df['Model']
        scores = results_df[metric]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = plt.bar(models, scores, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title(title or f'Model Comparison - {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'model_comparison.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title=None, save=True):
        """
        رسم نمودار باقیمانده‌ها برای رگرسیون
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Residual Analysis', fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'residual_analysis.jpg')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ نمودار در {save_path} ذخیره شد")
        
        plt.show()
        
        return residuals
    
    def create_summary_dashboard(self, df, target_col, task_type='classification'):
        """
        ایجاد داشبورد خلاصه از داده‌ها
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        ax1 = plt.subplot(3, 3, 1)
        if task_type == 'classification':
            df[target_col].value_counts().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title('Target Class Distribution')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
        else:
            ax1.hist(df[target_col], bins=30, edgecolor='black', alpha=0.7)
            ax1.set_title('Target Distribution')
            ax1.set_xlabel(target_col)
            ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Missing values
        ax2 = plt.subplot(3, 3, 2)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing.plot(kind='bar', ax=ax2, color='salmon', edgecolor='black')
            ax2.set_title('Missing Values by Column')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values')
        ax2.grid(True, alpha=0.3)
        
        # 3. Data types
        ax3 = plt.subplot(3, 3, 3)
        dtypes = df.dtypes.value_counts()
        dtypes.plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_title('Data Types')
        ax3.set_xlabel('Type')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. Numerical features distributions (first 4)
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numerical_cols):
            ax = plt.subplot(3, 3, 4 + i)
            ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 5. Correlation heatmap (if enough numerical columns)
        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
            ax8 = plt.subplot(3, 3, 8)
            corr = df.select_dtypes(include=[np.number]).corr()
            im = ax8.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax8.set_title('Correlation Heatmap')
            ax8.set_xticks(range(len(corr.columns)))
            ax8.set_yticks(range(len(corr.columns)))
            ax8.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax8.set_yticklabels(corr.columns, fontsize=8)
            plt.colorbar(im, ax=ax8)
        
        # 6. Categorical features (top categories of first categorical column)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            ax9 = plt.subplot(3, 3, 9)
            top_cat = df[categorical_cols[0]].value_counts().head(10)
            top_cat.plot(kind='bar', ax=ax9, color='purple', edgecolor='black', alpha=0.7)
            ax9.set_title(f'Top Categories: {categorical_cols[0]}')
            ax9.set_xlabel(categorical_cols[0])
            ax9.set_ylabel('Count')
            ax9.tick_params(axis='x', rotation=45)
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Data Summary Dashboard', fontsize=20, y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.output_dir, 'data_dashboard.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard در {save_path} ذخیره شد")
        
        plt.show()