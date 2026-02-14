"""
پایپ‌لاین اصلی پروژه مدلسازی داده‌های کنکور ایران
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings('ignore')

# تنظیمات محیط برای reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False

# تنظیم seedها برای reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)

# Import custom modules
from exam_data_manager import ExamDataManager, ExamDataAnalyzer
from exam_trainer import ExamModelTrainer
from exam_evaluator import ExamModelEvaluator
from exam_utils import ExamModelUtils


class IranExamPipeline:
    """
    کلاس اصلی پایپ‌لاین پروژه کنکور ایران
    """

    def __init__(self, args):
        """
        Initialize pipeline with arguments

        Parameters:
        -----------
        args : dict or DotDict
            Pipeline arguments
        """
        self.args = args
        self.root_dir = Path(__file__).parent
        self.data_manager = None
        self.trainer = None
        self.evaluator = None

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_logging()

        print("🎓 پایپ‌لاین مدلسازی کنکور ایران راه‌اندازی شد")
        print("="*60)

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.args.get('data_path', 'data'),
            self.args.get('models_path', 'models'),
            self.args.get('plots_path', 'plots'),
            self.args.get('results_path', 'results'),
            self.args.get('logs_path', 'logs')
        ]

        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 پوشه ایجاد شد: {dir_path}")

    def setup_logging(self):
        """Setup logging and recording"""
        self.recording_file = self.root_dir / self.args.get('recording_file', 'pipeline_output.txt')

        # Clear previous recording if requested
        if self.args.get('clear_previous', True):
            with open(self.recording_file, 'w', encoding='utf-8') as f:
                f.write("📝 گزارش اجرای پایپ‌لاین کنکور ایران\n")
                f.write("="*60 + "\n\n")

        print(f"📝 فایل گزارش: {self.recording_file}")

    def log_message(self, message):
        """Log message to file and console"""
        print(message)
        with open(self.recording_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def run_data_pipeline(self):
        """
        اجرای کامل پایپ‌لاین داده‌ها
        """
        self.log_message("\n📊 شروع پایپ‌لاین داده‌ها")
        self.log_message("="*60)

        # 1. ایجاد مدیر داده
        self.log_message("1. ایجاد مدیر داده...")
        self.data_manager = ExamDataManager(
            data_dir=str(self.root_dir / self.args.get('data_path', 'data')),
            recording_file=str(self.recording_file),
            plots_folder=str(self.root_dir / self.args.get('plots_path', 'plots'))
        )

        # 2. بارگذاری و آماده‌سازی داده‌ها
        self.log_message("2. بارگذاری و آماده‌سازی داده‌ها...")
        df = self.data_manager.load_and_prepare_data(
            data_path=self.args.get('exam_data_path', 'data/iran_exam.csv'),
            task_type=self.args.get('task_type', 'classification')
        )

        # 3. تحلیل اکتشافی داده‌ها
        self.log_message("3. تحلیل اکتشافی داده‌ها...")
        self.data_manager.exploratory_data_analysis()

        # 4. آماده‌سازی برای مدل‌های مختلف
        self.log_message("4. آماده‌سازی برای مدل‌ها...")
        if 'TabTransformer' in self.args.get('models', []):
            self.data_manager.prepare_for_tabtransformer()

        self.data_manager.prepare_for_traditional_models()

        # 5. ایجاد تقسیم‌بندی داده‌ها
        self.log_message("5. ایجاد تقسیم‌بندی داده‌ها...")
        splits = self.data_manager.create_train_val_test_split(
            train_size=self.args.get('train_size', 0.7),
            val_size=self.args.get('val_size', 0.15),
            test_size=self.args.get('test_size', 0.15)
        )

        self.log_message("✅ پایپ‌لاین داده‌ها کامل شد")
        return df

    def run_training_pipeline(self):
        """
        اجرای پایپ‌لاین آموزش مدل‌ها
        """
        self.log_message("\n🎯 شروع پایپ‌لاین آموزش مدل‌ها")
        self.log_message("="*60)

        if self.data_manager is None:
            self.log_message("❌ ابتدا پایپ‌لاین داده‌ها را اجرا کنید")
            return None

        # 1. ایجاد Trainer
        self.log_message("1. ایجاد Trainer...")
        self.trainer = ExamModelTrainer(
            data_manager=self.data_manager,
            output_dir=str(self.root_dir / self.args.get('models_path', 'models')),
            random_state=self.args.get('random_state', 42)
        )

        # 2. آموزش مدل‌های انتخابی
        models_to_train = self.args.get('models', ['RandomForest', 'MLP', 'TabTransformer'])
        self.log_message(f"2. آموزش مدل‌ها: {models_to_train}")

        all_results = {}
        for model_type in models_to_train:
            try:
                self.log_message(f"  🎯 آموزش مدل: {model_type}")

                results = self.trainer.nested_cross_validation(
                    model_type=model_type,
                    k_outer=self.args.get('k_fold_cv', 5),
                    k_inner=self.args.get('k_inner_cv', 3)
                )

                all_results[model_type] = results
                self.log_message(f"  ✅ آموزش {model_type} کامل شد")

            except Exception as e:
                self.log_message(f"  ❌ خطا در آموزش {model_type}: {e}")

        # 3. مقایسه مدل‌ها
        self.log_message("3. مقایسه مدل‌ها...")
        comparison = self.trainer.compare_models(models_to_train)

        # ذخیره نتایج مقایسه
        comparison_path = self.root_dir / self.args.get('results_path', 'results') / 'model_comparison.csv'
        comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        self.log_message(f"  💾 نتایج مقایسه در {comparison_path} ذخیره شد")

        self.log_message("✅ پایپ‌لاین آموزش کامل شد")
        return all_results

    def run_evaluation_pipeline(self):
        """
        اجرای پایپ‌لاین ارزیابی مدل‌ها
        """
        self.log_message("\n🧪 شروع پایپ‌لاین ارزیابی مدل‌ها")
        self.log_message("="*60)

        if self.data_manager is None:
            self.log_message("❌ ابتدا پایپ‌لاین داده‌ها را اجرا کنید")
            return None

        # 1. ایجاد Evaluator
        self.log_message("1. ایجاد Evaluator...")
        self.evaluator = ExamModelEvaluator(
            data_manager=self.data_manager,
            output_dir=str(self.root_dir / self.args.get('results_path', 'results'))
        )

        # 2. ارزیابی همه مدل‌های ذخیره شده
        self.log_message("2. ارزیابی مدل‌های ذخیره شده...")
        evaluation_results = self.evaluator.evaluate_all_models(
            models_dir=str(self.root_dir / self.args.get('models_path', 'models')),
            output_dir=str(self.root_dir / self.args.get('results_path', 'results'))
        )

        self.log_message("✅ پایپ‌لاین ارزیابی کامل شد")
        return evaluation_results

    def run_complete_pipeline(self):
        """
        اجرای کامل پایپ‌لاین از ابتدا تا انتها
        """
        self.log_message("\n🚀 شروع اجرای کامل پایپ‌لاین")
        self.log_message("="*60)

        start_time = pd.Timestamp.now()
        self.log_message(f"⏰ زمان شروع: {start_time}")

        # اجرای مراحل
        try:
            # 1. پایپ‌لاین داده‌ها
            df = self.run_data_pipeline()

            # 2. پایپ‌لاین آموزش
            training_results = self.run_training_pipeline()

            # 3. پایپ‌لاین ارزیابی
            evaluation_results = self.run_evaluation_pipeline()

            end_time = pd.Timestamp.now()
            duration = end_time - start_time

            self.log_message(f"\n✅ پایپ‌لاین با موفقیت کامل شد!")
            self.log_message(f"⏱️  مدت زمان اجرا: {duration}")

            return {
                'data': df,
                'training': training_results,
                'evaluation': evaluation_results,
                'duration': duration
            }

        except Exception as e:
            self.log_message(f"\n❌ خطا در اجرای پایپ‌لاین: {e}")
            import traceback
            traceback.print_exc()
            return None


class DotDict(dict):
    """دیکشنری با دسترسی dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_exam_args():
    """
    تنظیم پارامترهای پایپ‌لاین کنکور
    """
    args = {
        # مسیرهای داده
        'exam_data_path': 'data/iran_exam.csv',
        'data_path': 'exam_data',
        'models_path': 'exam_models',
        'plots_path': 'exam_plots',
        'results_path': 'exam_results',
        'logs_path': 'exam_logs',

        # فایل‌های خروجی
        'recording_file': 'exam_pipeline_output.txt',

        # تنظیمات پایپ‌لاین
        'clear_previous': True,
        'random_state': 42,
        'task_type': 'classification',  # یا 'regression'

        # تقسیم‌بندی داده‌ها
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,

        # مدل‌ها برای آموزش
        'models': ['RandomForest', 'XGBoost', 'LightGBM', 'MLP', 'TabTransformer'],

        # تنظیمات Cross-Validation
        'k_fold_cv': 5,
        'k_inner_cv': 3,

        # تنظیمات اضافی
        'use_gpu': True,
        'verbose': True,
        'save_all_models': True
    }

    return DotDict(args)


def main():
    """
    تابع اصلی اجرای پایپ‌لاین
    """
    print("🎓 پروژه مدلسازی داده‌های کنکور ایران")
    print("="*60)

    # دریافت آرگومان‌ها
    args = get_exam_args()

    # ایجاد و اجرای پایپ‌لاین
    pipeline = IranExamPipeline(args)

    # اجرای کامل پایپ‌لاین
    results = pipeline.run_complete_pipeline()

    if results:
        print("\n📊 خلاصه نتایج:")
        if results.get('training'):
            for model_type, model_results in results['training'].items():
                if 'mean_test_score' in model_results:
                    print(f"   {model_type}: {model_results['mean_test_score']:.4f}")

    print("\n✅ اجرای کامل پروژه با موفقیت به پایان رسید!")
    print("="*60)


if __name__ == '__main__':
    main()