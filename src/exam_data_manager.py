"""
مدیریت داده‌های کنکور ایران
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class ExamDataManager:
    """
    کلاس مدیریت داده‌های کنکور
    """
    
    def __init__(self, data_dir='data', recording_file='analysis.txt', plots_folder='plots'):
        """
        Initialize data manager
        
        Parameters:
        -----------
        data_dir : str
            Directory containing exam data
        recording_file : str
            File to record analysis results
        plots_folder : str
            Folder to save plots
        """
        self.data_dir = data_dir
        self.recording_file = recording_file
        self.plots_folder = plots_folder
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)
        
        # Data attributes
        self.df = None
        self.X = None
        self.y = None
        self.num_classes = None
        self.feature_names = None
        self.target_col = None
        self.task_type = None
        
        # For TabTransformer
        self.categories = None
        self.continuous_features = 0
        self.X_cat = None
        self.X_cont = None
        
        # For splits
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        print(f"📁 مدیر داده‌ها ایجاد شد: data_dir={data_dir}")
    
    def load_and_prepare_data(self, data_path=None, task_type='classification'):
        """
        Load and prepare exam data
        
        Parameters:
        -----------
        data_path : str, optional
            Path to exam data file
        task_type : str
            Type of task: 'classification' or 'regression'
        
        Returns:
        --------
        pd.DataFrame: Loaded and prepared data
        """
        print("\n🎓 در حال بارگذاری و آماده‌سازی داده‌های کنکور...")
        
        self.task_type = task_type
        
        # If no specific path, try different sources
        if data_path is None:
            data_path = self._find_exam_data()
        
        # Load data
        self._load_exam_data(data_path)
        
        # Clean and preprocess
        self._clean_data()
        
        # Define task
        self._define_task()
        
        print(f"✅ داده‌ها بارگذاری شدند: {len(self.df)} نمونه، {len(self.df.columns)} ویژگی")
        
        return self.df
    
    def _find_exam_data(self):
        """Try to find exam data from different sources"""
        possible_paths = [
            'iran_exam.csv',
            'data/iran_exam.csv',
            '/content/iran_exam.csv',
            'exam_data.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 یافتن داده‌ها در: {path}")
                return path
        
        raise FileNotFoundError("❌ فایل داده‌های کنکور یافت نشد")
    
    def _load_exam_data(self, data_path):
        """Load exam data from CSV file"""
        try:
            self.df = pd.read_csv(data_path)
            print(f"📊 داده‌ها بارگذاری شدند. شکل: {self.df.shape}")
            
            # Display basic info
            print("\n📋 اطلاعات اولیه:")
            print(f"  ستون‌ها: {list(self.df.columns)}")
            print(f"  انواع داده‌ها:")
            for col, dtype in self.df.dtypes.items():
                print(f"    {col}: {dtype}")
            print(f"  مقادیر گمشده: {self.df.isnull().sum().sum()}")
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            raise
    
    def _clean_data(self):
        """Clean and preprocess exam data"""
        print("\n🧹 در حال پاکسازی داده‌ها...")
        
        # Save original column names for reference
        self.original_columns = self.df.columns.tolist()
        
        # 1. Handle missing values
        self._handle_missing_values()
        
        # 2. Standardize categorical columns
        self._standardize_categorical()
        
        # 3. Remove outliers (optional)
        self._remove_outliers()
        
        print("✅ پاکسازی داده‌ها انجام شد")
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"  🔍 مقادیر گمشده: {missing_before}")
            
            # Separate numeric and categorical columns
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            # Handle numeric columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"    {col}: پر کردن با میانه ({median_val:.2f})")
            
            # Handle categorical columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"    {col}: پر کردن با مد ({mode_val})")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  ✅ مقادیر گمشده باقی‌مانده: {missing_after}")
    
    def _standardize_categorical(self):
        """Standardize categorical values"""
        print("  🏷️ استانداردسازی مقادیر دسته‌ای...")
        
        # Standardize region values
        if 'منطقه' in self.df.columns:
            region_mapping = {
                'منطقه1': 'منطقه1', 'منطقهيک': 'منطقه1', 'منطقهیک': 'منطقه1', 'منطقه 1': 'منطقه1',
                'منطقه2': 'منطقه2', 'منطقهدو': 'منطقه2', 'منطقه 2': 'منطقه2',
                'منطقه3': 'منطقه3', 'منطقهسه': 'منطقه3', 'منطقه 3': 'منطقه3'
            }
            
            self.df['منطقه'] = self.df['منطقه'].apply(
                lambda x: region_mapping.get(str(x).strip(), str(x).strip())
            )
            print(f"    منطقه: استانداردسازی {self.df['منطقه'].nunique()} مقدار یکتا")
        
        # Standardize city names (remove extra spaces)
        if 'شهر' in self.df.columns:
            self.df['شهر'] = self.df['شهر'].str.strip().str.replace(r'\s+', ' ', regex=True)
            print(f"    شهر: {self.df['شهر'].nunique()} شهر یکتا")
    
    def _remove_outliers(self, method='iqr', threshold=3):
        """Remove outliers from numerical columns"""
        if method == 'iqr':
            numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            
            for col in numerical_cols:
                if col != 'رتبه کشوری':  # Don't remove outliers from target
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    before = len(self.df)
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    after = len(self.df)
                    
                    if before > after:
                        print(f"    {col}: حذف {before - after} outlier")
    
    def _define_task(self):
        """Define the prediction task"""
        print(f"\n🎯 تعریف وظیفه: {self.task_type}")
        
        if self.task_type == 'classification':
            # Create binary classification task (top 20% vs others)
            if 'رتبه کشوری' in self.df.columns:
                threshold = self.df['رتبه کشوری'].quantile(0.2)  # Top 20%
                self.df['target'] = (self.df['رتبه کشوری'] <= threshold).astype(int)
                self.target_col = 'target'
                print(f"  طبقه‌بندی: قبولی در 20% برتر (آستانه: رتبه ≤ {threshold:.0f})")
                
                # Show class distribution
                class_counts = self.df['target'].value_counts()
                print(f"  توزیع کلاس‌ها: کلاس 0={class_counts.get(0, 0)}, کلاس 1={class_counts.get(1, 0)}")
        
        elif self.task_type == 'regression':
            # Use rank as continuous target
            if 'رتبه کشوری' in self.df.columns:
                self.target_col = 'رتبه کشوری'
                print(f"  رگرسیون: پیش‌بینی رتبه کشوری")
        
        else:
            raise ValueError(f"نوع وظیفه نامعتبر: {self.task_type}")
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        if self.df is None:
            print("❌ ابتدا داده‌ها را بارگذاری کنید")
            return
        
        print("\n🔍 شروع تحلیل اکتشافی داده‌ها...")
        
        # Create EDA results directory
        eda_dir = os.path.join(self.plots_folder, 'eda')
        os.makedirs(eda_dir, exist_ok=True)
        
        # 1. Basic information
        self._analyze_basic_info()
        
        # 2. Target variable analysis
        self._analyze_target()
        
        # 3. Numerical features
        self._analyze_numerical_features(eda_dir)
        
        # 4. Categorical features
        self._analyze_categorical_features(eda_dir)
        
        # 5. Correlation analysis
        self._analyze_correlations(eda_dir)
        
        print(f"✅ تحلیل اکتشافی کامل شد. نتایج در '{eda_dir}' ذخیره شد")
    
    def _analyze_basic_info(self):
        """Analyze basic information about the dataset"""
        with open(self.recording_file, 'a', encoding='utf-8') as f:
            f.write("\n📊 اطلاعات پایه داده‌ها\n")
            f.write("="*50 + "\n")
            f.write(f"تعداد نمونه‌ها: {len(self.df)}\n")
            f.write(f"تعداد ویژگی‌ها: {len(self.df.columns)}\n")
            f.write(f"ویژگی‌های عددی: {len(self.df.select_dtypes(include=['int64', 'float64']).columns)}\n")
            f.write(f"ویژگی‌های دسته‌ای: {len(self.df.select_dtypes(include=['object']).columns)}\n")
            f.write(f"مقادیر گمشده: {self.df.isnull().sum().sum()}\n")
    
    def _analyze_target(self):
        """Analyze target variable"""
        plt.figure(figsize=(12, 5))
        
        if self.task_type == 'classification':
            class_counts = self.df[self.target_col].value_counts()
            
            plt.subplot(1, 2, 1)
            plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'lightcoral'])
            plt.title('توزیع کلاس‌های هدف')
            plt.xlabel('کلاس')
            plt.ylabel('تعداد')
            plt.xticks([0, 1], ['کلاس 0', 'کلاس 1'])
            
            plt.subplot(1, 2, 2)
            plt.pie(class_counts.values, labels=['کلاس 0', 'کلاس 1'], 
                   autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
            plt.title('نسبت کلاس‌ها')
            
        else:  # regression
            plt.subplot(1, 2, 1)
            plt.hist(self.df[self.target_col], bins=50, edgecolor='black', alpha=0.7)
            plt.title('توزیع متغیر هدف')
            plt.xlabel(self.target_col)
            plt.ylabel('تعداد')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.df[self.target_col])
            plt.title('Boxplot متغیر هدف')
            plt.ylabel(self.target_col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'eda', 'target_analysis.jpg'), dpi=300)
        plt.close()
    
    def _analyze_numerical_features(self, save_dir):
        """Analyze numerical features"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_col]
        
        if len(numerical_cols) == 0:
            return
        
        # Distribution plots
        n_cols = min(len(numerical_cols), 6)
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        for i, col in enumerate(numerical_cols[:n_cols]):
            # Histogram
            axes[0, i].hist(self.df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[0, i].set_title(f'توزیع {col}')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('تعداد')
            
            # Boxplot
            axes[1, i].boxplot(self.df[col])
            axes[1, i].set_title(f'Boxplot {col}')
            axes[1, i].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'numerical_features.jpg'), dpi=300)
        plt.close()
    
    def _analyze_categorical_features(self, save_dir):
        """Analyze categorical features"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols[:3]:  # Limit to first 3
            plt.figure(figsize=(10, 6))
            
            # Get top 15 categories
            value_counts = self.df[col].value_counts().head(15)
            
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'توزیع {col} (15 دسته برتر)')
            plt.xlabel(col)
            plt.ylabel('تعداد')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'categorical_{col}.jpg'), dpi=300)
            plt.close()
    
    def _analyze_correlations(self, save_dir):
        """Analyze correlations between numerical features"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) < 2:
            return
        
        # Correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('ماتریس همبستگی ویژگی‌های عددی')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation_heatmap.jpg'), dpi=300)
        plt.close()
    
    def prepare_for_traditional_models(self):
        """
        Prepare data for traditional ML models (MLP, Random Forest, etc.)
        """
        print("\n🔄 آماده‌سازی داده برای مدل‌های سنتی...")
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        # Handle categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col in feature_cols]
        
        # Create feature matrix
        X_list = []
        
        # Numerical features
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        if numerical_cols:
            X_numerical = self.df[numerical_cols].values
            X_numerical = self.scaler.fit_transform(X_numerical)
            X_list.append(X_numerical)
        
        # Categorical features
        if categorical_cols:
            X_categorical_list = []
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_categorical_list.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le
            
            X_categorical = np.hstack(X_categorical_list)
            X_list.append(X_categorical)
        
        # Combine features
        if len(X_list) > 1:
            self.X = np.hstack(X_list)
        else:
            self.X = X_list[0]
        
        # Target
        self.y = self.df[self.target_col].values
        
        if self.task_type == 'classification':
            self.num_classes = len(np.unique(self.y))
        
        self.feature_names = feature_cols
        
        print(f"✅ داده‌ها آماده شدند: X shape: {self.X.shape}")
        
        return self.X, self.y
    
    def prepare_for_tabtransformer(self):
        """
        Prepare data for TabTransformer model
        """
        print("\n🔄 آماده‌سازی داده برای TabTransformer...")
        
        # Separate categorical and numerical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = [col for col in self.df.columns 
                         if col not in categorical_cols and col != self.target_col]
        
        # Prepare categorical features
        if len(categorical_cols) > 0:
            X_cat_list = []
            self.categories = []
            
            for col in categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                X_cat_list.append(encoded.reshape(-1, 1))
                self.categories.append(len(le.classes_))
                self.label_encoders[col] = le
            
            self.X_cat = np.hstack(X_cat_list).astype(np.int64)
        else:
            self.X_cat = np.zeros((len(self.df), 0), dtype=np.int64)
            self.categories = []
        
        # Prepare numerical features
        if len(numerical_cols) > 0:
            X_cont = self.df[numerical_cols].values.astype(np.float32)
            self.X_cont = self.scaler.fit_transform(X_cont)
            self.continuous_features = len(numerical_cols)
        else:
            self.X_cont = np.zeros((len(self.df), 0), dtype=np.float32)
            self.continuous_features = 0
        
        # Target
        self.y = self.df[self.target_col].values
        
        if self.task_type == 'classification':
            self.num_classes = len(np.unique(self.y))
        
        print(f"✅ داده‌ها برای TabTransformer آماده شدند:")
        print(f"  X_cat shape: {self.X_cat.shape}, categories: {self.categories}")
        print(f"  X_cont shape: {self.X_cont.shape}")
        print(f"  y shape: {self.y.shape}")
        
        return self.X_cat, self.X_cont, self.y
    
    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Create train/validation/test splits
        """
        print(f"\n✂️ ایجاد تقسیم‌بندی داده‌ها...")
        
        # First split: train vs temp
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            self.X, self.y, np.arange(len(self.X)),
            test_size=(val_size + test_size),
            random_state=42,
            stratify=self.y if self.task_type == 'classification' else None
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=test_size/(val_size+test_size),
            random_state=42,
            stratify=y_temp if self.task_type == 'classification' else None
        )
        
        # Store indices
        self.train_indices = idx_train
        self.val_indices = idx_val
        self.test_indices = idx_test
        
        # Store splits
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"✅ تقسیم‌بندی ایجاد شد:")
        print(f"  Train: {len(X_train)} نمونه")
        print(f"  Validation: {len(X_val)} نمونه")
        print(f"  Test: {len(X_test)} نمونه")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_tsne_visualization(self, features=None, n_samples=1000):
        """
        Create t-SNE visualization of high-dimensional data
        """
        if self.X is None:
            print("❌ ابتدا داده‌ها را آماده کنید")
            return
        
        print("\n🎨 ایجاد نمایش t-SNE...")
        
        # Sample data if too large
        if len(self.X) > n_samples:
            indices = np.random.choice(len(self.X), n_samples, replace=False)
            X_sample = self.X[indices]
            y_sample = self.y[indices]
        else:
            X_sample = self.X
            y_sample = self.y
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        if self.task_type == 'classification':
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=y_sample, cmap='tab10', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Class')
        else:
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=y_sample, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Target Value')
        
        plt.title('t-SNE Visualization of Exam Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_folder, 'tsne_visualization.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ t-SNE visualization saved to {save_path}")


class ExamDataAnalyzer:
    """
    کلاس تحلیل اکتشافی داده‌های کنکور
    """
    
    def __init__(self, df, output_dir='eda_results'):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        print("\n📊 تحلیل ویژگی‌های عددی")
        print("="*50)
        
        for col in self.numeric_cols[:6]:  # Limit to first 6
            print(f"\n📈 تحلیل ستون: {col}")
            print(f"   حداقل: {self.df[col].min():.2f}")
            print(f"   حداکثر: {self.df[col].max():.2f}")
            print(f"   میانگین: {self.df[col].mean():.2f}")
            print(f"   میانه: {self.df[col].median():.2f}")
            print(f"   انحراف معیار: {self.df[col].std():.2f}")
    
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        print("\n📊 تحلیل ویژگی‌های دسته‌ای")
        print("="*50)
        
        for col in self.categorical_cols[:5]:  # Limit to first 5
            print(f"\n🏷️ تحلیل ستون: {col}")
            value_counts = self.df[col].value_counts().head(10)
            print(f"   تعداد مقادیر یکتا: {len(value_counts)}")
            print("   10 مقدار برتر:")
            for val, count in value_counts.items():
                print(f"     {val}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def analyze_correlation(self):
        """Analyze correlation between numerical features"""
        if len(self.numeric_cols) < 2:
            return
        
        print("\n📈 تحلیل ماتریس همبستگی")
        print("="*50)
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                     corr_matrix.iloc[i, j]))
        
        if high_corr:
            print("همبستگی‌های قوی (> 0.7):")
            for f1, f2, corr in high_corr:
                print(f"  {f1} <-> {f2}: {corr:.3f}")
    
    def analyze_missing_values(self):
        """Analyze missing values"""
        print("\n🔍 تحلیل مقادیر گمشده")
        print("="*50)
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        }).sort_values('Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing Values'] > 0])
