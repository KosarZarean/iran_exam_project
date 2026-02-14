"""
توابع کمکی برای مدیریت مدل‌های کنکور ایران
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExamModelUtils:
    """
    کلاس توابع کمکی برای مدیریت مدل‌های کنکور
    """
    
    @staticmethod
    def create_model_name(model_type: str, config: Dict[str, Any], 
                         fold: Optional[int] = None, version: str = 'v1') -> str:
        """
        ایجاد نام معنی‌دار برای مدل
        
        Parameters:
        -----------
        model_type : str
            نوع مدل
        config : Dict[str, Any]
            پیکربندی مدل
        fold : int, optional
            شماره fold
        version : str
            نسخه مدل
        
        Returns:
        --------
        str
            نام مدل
        """
        # نام پایه
        if fold is not None:
            base_name = f"{model_type}_fold{fold}_{version}"
        else:
            base_name = f"{model_type}_{version}"
        
        # اضافه کردن پارامترهای مهم
        param_parts = []
        
        if model_type == 'MLP':
            important_params = ['hidden_layers', 'dropout_rate', 'learning_rate', 'batch_size']
        elif model_type == 'TabTransformer':
            important_params = ['embedding_dim', 'num_heads', 'num_layers', 'transformer_dropout']
        elif model_type == 'RandomForest':
            important_params = ['n_estimators', 'max_depth', 'min_samples_split']
        elif model_type == 'XGBoost':
            important_params = ['n_estimators', 'max_depth', 'learning_rate']
        elif model_type == 'LightGBM':
            important_params = ['n_estimators', 'num_leaves', 'learning_rate']
        elif model_type == 'SVM':
            important_params = ['C', 'kernel', 'gamma']
        elif model_type == 'Logistic':
            important_params = ['C', 'penalty', 'solver']
        else:
            important_params = []
        
        for param in important_params:
            if param in config:
                value = config[param]
                if isinstance(value, (list, tuple)):
                    value = '_'.join(str(v) for v in value)
                param_parts.append(f"{param[:3]}_{str(value).replace('.', '')}")
        
        # اضافه کردن timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # ترکیب همه بخش‌ها
        if param_parts:
            model_name = f"{base_name}_{'_'.join(param_parts)}_{timestamp}"
        else:
            model_name = f"{base_name}_{timestamp}"
        
        return model_name
    
    @staticmethod
    def save_model(model: Any, model_path: str, model_type: str, 
                  config: Dict[str, Any], scaler: Optional[Any] = None,
                  label_encoder: Optional[Any] = None, feature_names: Optional[List[str]] = None,
                  metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        ذخیره مدل و اطلاعات مرتبط
        
        Parameters:
        -----------
        model : Any
            مدل آموزش دیده
        model_path : str
            مسیر ذخیره مدل
        model_type : str
            نوع مدل
        config : Dict[str, Any]
            پیکربندی مدل
        scaler : Any, optional
            شیء scaler
        label_encoder : Any, optional
            شیء label encoder
        feature_names : List[str], optional
            نام ویژگی‌ها
        metrics : Dict[str, float], optional
            معیارهای ارزیابی
        
        Returns:
        --------
        Dict[str, Any]
            اطلاعات ذخیره شده
        """
        # ایجاد پوشه اگر وجود ندارد
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        save_info = {
            'model_type': model_type,
            'config': config,
            'save_time': pd.Timestamp.now().isoformat(),
            'feature_names': feature_names,
            'metrics': metrics
        }
        
        # ذخیره بر اساس نوع مدل
        if model_type in ['MLP', 'TabTransformer', 'Regressor']:
            # ذخیره مدل PyTorch
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_type': model_type,
                'save_info': save_info
            }, model_path)
            
            # ذخیره save_info جداگانه
            info_path = model_path.replace('.pt', '_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(save_info, f, ensure_ascii=False, indent=2)
        
        else:
            # ذخیره مدل scikit-learn
            import joblib
            joblib.dump(model, model_path)
            
            # ذخیره save_info
            info_path = model_path.replace('.pkl', '_info.json').replace('.joblib', '_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(save_info, f, ensure_ascii=False, indent=2)
        
        # ذخیره scaler اگر وجود دارد
        if scaler is not None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl').replace('.pkl', '_scaler.pkl').replace('.joblib', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            save_info['scaler_path'] = scaler_path
        
        # ذخیره label encoder اگر وجود دارد
        if label_encoder is not None:
            le_path = model_path.replace('.pt', '_label_encoder.pkl').replace('.pkl', '_label_encoder.pkl').replace('.joblib', '_label_encoder.pkl')
            with open(le_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            save_info['label_encoder_path'] = le_path
        
        print(f"✅ مدل در {model_path} ذخیره شد")
        
        return save_info
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        بارگذاری مدل و اطلاعات مرتبط
        
        Parameters:
        -----------
        model_path : str
            مسیر مدل
        
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            مدل و اطلاعات مرتبط
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"فایل مدل یافت نشد: {model_path}")
        
        # بارگذاری اطلاعات مدل
        info_path = model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                save_info = json.load(f)
        else:
            save_info = {}
        
        # بارگذاری مدل بر اساس پسوند
        if model_path.endswith('.pt'):
            # بارگذاری مدل PyTorch
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # ایجاد مدل بر اساس type
            model_type = checkpoint.get('model_type', 'MLP')
            config = checkpoint.get('config', {})
            
            if model_type == 'MLP':
                from exam_models import ExamMLP
                input_dim = config.get('input_dim', 10)
                hidden_dims = config.get('hidden_dims', (128, 64))
                num_classes = config.get('num_classes', 2)
                model = ExamMLP(input_dim, hidden_dims, num_classes)
            elif model_type == 'TabTransformer':
                from exam_models import ExamTabTransformer
                num_categorical = config.get('num_categorical', 3)
                num_continuous = config.get('num_continuous', 7)
                categories = config.get('categories', [10, 5, 8])
                num_classes = config.get('num_classes', 2)
                model = ExamTabTransformer(num_categorical, num_continuous, categories, num_classes)
            elif model_type == 'Regressor':
                from exam_models import ExamRegressor
                input_dim = config.get('input_dim', 10)
                hidden_dims = config.get('hidden_dims', (128, 64))
                model = ExamRegressor(input_dim, hidden_dims)
            else:
                raise ValueError(f"نوع مدل نامعتبر: {model_type}")
            
            # بارگذاری وزن‌ها
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        else:
            # بارگذاری مدل scikit-learn
            import joblib
            model = joblib.load(model_path)
        
        # بارگذاری scaler اگر وجود دارد
        scaler = None
        scaler_path = model_path.replace('.pt', '_scaler.pkl').replace('.pkl', '_scaler.pkl').replace('.joblib', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            save_info['scaler'] = scaler
        
        # بارگذاری label encoder اگر وجود دارد
        le_path = model_path.replace('.pt', '_label_encoder.pkl').replace('.pkl', '_label_encoder.pkl').replace('.joblib', '_label_encoder.pkl')
        if os.path.exists(le_path):
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
            save_info['label_encoder'] = label_encoder
        
        print(f"✅ مدل از {model_path} بارگذاری شد")
        
        return model, save_info
    
    @staticmethod
    def remove_old_models(models_dir: str, model_type: str, keep_best: int = 3, 
                         metric: str = 'roc_auc_ovr') -> List[str]:
        """
        حذف مدل‌های قدیمی و نگه‌داری بهترین‌ها
        
        Parameters:
        -----------
        models_dir : str
            پوشه مدل‌ها
        model_type : str
            نوع مدل
        keep_best : int
            تعداد بهترین مدل‌ها برای نگه‌داری
        metric : str
            معیار برای انتخاب بهترین‌ها
        
        Returns:
        --------
        List[str]
            لیست مدل‌های حذف شده
        """
        # پیدا کردن همه مدل‌های از نوع مشخص
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith(model_type) and (file.endswith('.pt') or file.endswith('.pkl') or file.endswith('.joblib')):
                model_files.append(file)
        
        if len(model_files) <= keep_best:
            print(f"⚠️ تنها {len(model_files)} مدل {model_type} وجود دارد، هیچ مدلی حذف نمی‌شود")
            return []
        
        # جمع‌آوری معیارهای هر مدل
        model_metrics = []
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            info_path = model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json')
            
            score = 0
            save_time = ''
            
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                metrics = info.get('metrics', {})
                score = metrics.get(metric, 0)
                save_time = info.get('save_time', '')
            
            model_metrics.append((model_file, score, save_time))
        
        # مرتب‌سازی بر اساس معیار (نزولی)
        model_metrics.sort(key=lambda x: x[1], reverse=True)
        
        # جدا کردن مدل‌های برای نگه‌داری و حذف
        keep_models = [m[0] for m in model_metrics[:keep_best]]
        remove_models = [m[0] for m in model_metrics[keep_best:]]
        
        # حذف مدل‌های قدیمی
        removed = []
        for model_file in remove_models:
            model_path = os.path.join(models_dir, model_file)
            
            # حذف فایل‌های مرتبط
            related_files = [
                model_path,
                model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json'),
                model_path.replace('.pt', '_scaler.pkl').replace('.pkl', '_scaler.pkl').replace('.joblib', '_scaler.pkl'),
                model_path.replace('.pt', '_label_encoder.pkl').replace('.pkl', '_label_encoder.pkl').replace('.joblib', '_label_encoder.pkl')
            ]
            
            for file_path in related_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        removed.append(os.path.basename(file_path))
                    except Exception as e:
                        print(f"❌ خطا در حذف {file_path}: {e}")
        
        print(f"🗑️  {len(removed)} فایل حذف شدند. {keep_best} مدل برتر نگه‌داری شدند")
        
        return removed
    
    @staticmethod
    def find_best_model(models_dir: str, model_type: str, 
                       metric: str = 'roc_auc_ovr') -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        پیدا کردن بهترین مدل بر اساس معیار مشخص
        
        Parameters:
        -----------
        models_dir : str
            پوشه مدل‌ها
        model_type : str
            نوع مدل
        metric : str
            معیار برای انتخاب بهترین
        
        Returns:
        --------
        Tuple[Optional[str], Optional[Dict[str, Any]]]
            مسیر بهترین مدل و اطلاعات آن
        """
        # پیدا کردن همه مدل‌های از نوع مشخص
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith(model_type) and (file.endswith('.pt') or file.endswith('.pkl') or file.endswith('.joblib')):
                model_files.append(file)
        
        if not model_files:
            print(f"⚠️ هیچ مدل {model_type} یافت نشد")
            return None, None
        
        # بررسی معیارهای هر مدل
        best_model = None
        best_score = -np.inf
        best_info = None
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            info_path = model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json')
            
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                metrics = info.get('metrics', {})
                score = metrics.get(metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_path
                    best_info = info
        
        if best_model:
            print(f"🏆 بهترین مدل {model_type}: {os.path.basename(best_model)}")
            print(f"   معیار {metric}: {best_score:.4f}")
        
        return best_model, best_info
    
    @staticmethod
    def compare_models(models_dir: str, metric: str = 'roc_auc_ovr') -> pd.DataFrame:
        """
        مقایسه همه مدل‌های موجود
        
        Parameters:
        -----------
        models_dir : str
            پوشه مدل‌ها
        metric : str
            معیار برای مقایسه
        
        Returns:
        --------
        pd.DataFrame
            جدول مقایسه
        """
        # جمع‌آوری اطلاعات همه مدل‌ها
        model_data = []
        
        for file in os.listdir(models_dir):
            if file.endswith(('.pt', '.pkl', '.joblib')):
                model_path = os.path.join(models_dir, file)
                info_path = model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json')
                
                if os.path.exists(info_path):
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    
                    metrics = info.get('metrics', {})
                    
                    model_data.append({
                        'model_file': file,
                        'model_type': info.get('model_type', 'Unknown'),
                        'metric_score': metrics.get(metric, 0),
                        'save_time': info.get('save_time', ''),
                        'config': str(info.get('config', {}))
                    })
        
        # ایجاد DataFrame
        if model_data:
            df = pd.DataFrame(model_data)
            df = df.sort_values(by='metric_score', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    @staticmethod
    def export_model_for_production(model_path: str, output_dir: str) -> Dict[str, str]:
        """
        صادر کردن مدل برای استفاده در تولید
        
        Parameters:
        -----------
        model_path : str
            مسیر مدل اصلی
        output_dir : str
            پوشه خروجی
        
        Returns:
        --------
        Dict[str, str]
            مسیر فایل‌های صادر شده
        """
        # ایجاد پوشه خروجی
        os.makedirs(output_dir, exist_ok=True)
        
        # بارگذاری مدل
        model, info = ExamModelUtils.load_model(model_path)
        
        # کپی کردن فایل‌های مرتبط
        base_name = os.path.basename(model_path).split('.')[0]
        exported_files = {}
        
        # کپی فایل مدل
        if model_path.endswith('.pt'):
            exported_path = os.path.join(output_dir, f"{base_name}.pt")
            shutil.copy2(model_path, exported_path)
            exported_files['model'] = exported_path
        
        # کپی فایل‌های اطلاعاتی
        related_files = [
            model_path.replace('.pt', '_info.json').replace('.pkl', '_info.json').replace('.joblib', '_info.json'),
            model_path.replace('.pt', '_scaler.pkl').replace('.pkl', '_scaler.pkl').replace('.joblib', '_scaler.pkl'),
            model_path.replace('.pt', '_label_encoder.pkl').replace('.pkl', '_label_encoder.pkl').replace('.joblib', '_label_encoder.pkl')
        ]
        
        for src_file in related_files:
            if os.path.exists(src_file):
                dst_file = os.path.join(output_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dst_file)
                exported_files[os.path.basename(src_file).split('.')[0]] = dst_file
        
        # ایجاد فایل README
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# مدل {info.get('model_type', 'Unknown')}\n\n")
            f.write(f"## اطلاعات مدل\n")
            f.write(f"- نوع: {info.get('model_type', 'Unknown')}\n")
            f.write(f"- تاریخ ذخیره: {info.get('save_time', '')}\n")
            f.write(f"- معیارها: {json.dumps(info.get('metrics', {}), indent=2, ensure_ascii=False)}\n")
            f.write(f"\n## نحوه استفاده\n")
            f.write("```python\n")
            f.write("from exam_utils import ExamModelUtils\n")
            f.write("model, info = ExamModelUtils.load_model('model.pt')\n")
            f.write("```\n")
        
        exported_files['readme'] = readme_path
        
        print(f"✅ مدل برای تولید در {output_dir} صادر شد")
        
        return exported_files


def print_model_config(config: Dict[str, Any], title: str = "پیکربندی مدل") -> None:
    """
    چاپ پیکربندی مدل به صورت زیبا
    
    Parameters:
    -----------
    config : Dict[str, Any]
        پیکربندی مدل
    title : str
        عنوان نمایش
    """
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")
    
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            value_str = ', '.join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            value_str = str(value)
        
        print(f"  {key}: {value_str}")
    
    print(f"{'='*60}")


def save_training_results(results: Dict[str, Any], save_path: str) -> None:
    """
    ذخیره نتایج آموزش
    
    Parameters:
    -----------
    results : Dict[str, Any]
        نتایج آموزش
    save_path : str
        مسیر ذخیره
    """
    # ایجاد پوشه اگر وجود ندارد
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # ذخیره به فرمت JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ نتایج آموزش در {save_path} ذخیره شد")


def load_training_results(load_path: str) -> Dict[str, Any]:
    """
    بارگذاری نتایج آموزش
    
    Parameters:
    -----------
    load_path : str
        مسیر بارگذاری
    
    Returns:
    --------
    Dict[str, Any]
        نتایج آموزش
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"فایل نتایج یافت نشد: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ نتایج آموزش از {load_path} بارگذاری شد")
    
    return results