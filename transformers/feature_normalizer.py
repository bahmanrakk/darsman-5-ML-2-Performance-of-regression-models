import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNormalizerTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر جامع نرمال‌سازی و تبدیل توزیع ویژگی‌ها
    پشتیبانی از تبدیلات: sqrt، log، boxcox، standardization
    """
    
    def __init__(
        self,
        sqrt_columns: list = None,
        log_columns: list = None,
        boxcox_columns: list = None,
        standardize_directly: list = None,
        keep_original: bool = False,
        add_constant: float = 1e-5
    ):
        """
        پارامترها:
            sqrt_columns: لیست ستون‌هایی که تبدیل ریشه دوم می‌شوند
            log_columns: لیست ستون‌هایی که تبدیل لگاریتمی می‌شوند
            boxcox_columns: لیست ستون‌هایی که تبدیل باکس-کاکس می‌شوند
            standardize_directly: لیست ستون‌هایی که مستقیماً استاندارد می‌شوند
            keep_original: آیا ستون‌های اصلی حفظ شوند؟
            add_constant: مقدار ثابت برای جلوگیری از log(0) یا boxcox خطا
        """
        self.sqrt_columns = sqrt_columns or []
        self.log_columns = log_columns or []
        self.boxcox_columns = boxcox_columns or []
        self.standardize_directly = standardize_directly or []
        self.keep_original = keep_original
        self.add_constant = add_constant
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        print("\n" + "=" * 60)
        print("🔄 مرحله نرمال‌سازی و تبدیل توزیع ویژگی‌ها")
        print("=" * 60)
        
        # مرحله 1: تبدیل ریشه دوم
        if self.sqrt_columns:
            df = self._apply_sqrt(df, self.sqrt_columns)
        
        # مرحله 2: تبدیل لگاریتمی
        if self.log_columns:
            df = self._apply_log(df, self.log_columns)
        
        # مرحله 3: تبدیل باکس-کاکس
        if self.boxcox_columns:
            df = self._apply_boxcox(df, self.boxcox_columns)
        
        # مرحله 4: استانداردسازی مستقیم
        if self.standardize_directly:
            df = self._apply_standardization(df, self.standardize_directly)
        
        print("\n✅ تمام تبدیلات با موفقیت اعمال شدند")
        print(f"📊 تعداد ستون‌های نهایی: {len(df.columns)}")
        
        return df
    
    def _apply_sqrt(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """اعمال تبدیل ریشه دوم"""
        print(f"\n📐 اعمال تبدیل ریشه دوم روی: {columns}")
        
        for col in columns:
            if col not in df.columns:
                print(f"   ⚠️ ستون '{col}' یافت نشد")
                continue
            
            new_col = f"{col}_sqrt"
            df[new_col] = np.sqrt(df[col])
            
            if not self.keep_original:
                df = df.drop(columns=[col])
            
            print(f"   ✅ {col} → {new_col}")
        
        # استانداردسازی ستون‌های sqrt شده
        sqrt_std_columns = [f"{col}_sqrt" for col in columns if f"{col}_sqrt" in df.columns]
        if sqrt_std_columns:
            df = self._apply_standardization(df, sqrt_std_columns)
        
        return df
    
    def _apply_log(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """اعمال تبدیل لگاریتمی"""
        print(f"\n📊 اعمال تبدیل لگاریتمی روی: {columns}")
        
        for col in columns:
            if col not in df.columns:
                print(f"   ⚠️ ستون '{col}' یافت نشد")
                continue
            
            new_col = f"{col}_log"
            
            # بررسی مقادیر غیرمثبت
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                df[new_col] = np.log(df[col] + self.add_constant)
                print(f"   ⚠️ {col}: {non_positive} مقدار غیرمثبت → افزودن ثابت {self.add_constant}")
            else:
                df[new_col] = np.log(df[col])
            
            if not self.keep_original:
                df = df.drop(columns=[col])
            
            print(f"   ✅ {col} → {new_col}")
        
        # استانداردسازی ستون‌های log شده
        log_std_columns = [f"{col}_log" for col in columns if f"{col}_log" in df.columns]
        if log_std_columns:
            df = self._apply_standardization(df, log_std_columns)
        
        return df
    
    def _apply_boxcox(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """اعمال تبدیل باکس-کاکس"""
        print(f"\n📦 اعمال تبدیل باکس-کاکس روی: {columns}")
        
        for col in columns:
            if col not in df.columns:
                print(f"   ⚠️ ستون '{col}' یافت نشد")
                continue
            
            new_col = f"{col}_boxcox"
            
            # بررسی مقادیر غیرمثبت
            non_positive = (df[col] <= 0).sum()
            constant_used = 0
            
            if non_positive > 0:
                constant_used = abs(df[col].min()) + self.add_constant
                data_for_transform = df[col] + constant_used
                print(f"   ⚠️ {col}: {non_positive} مقدار غیرمثبت → افزودن ثابت {constant_used:.6f}")
            else:
                data_for_transform = df[col]
            
            try:
                transformed_data, lambda_val = stats.boxcox(data_for_transform)
                df[new_col] = transformed_data
                
                if not self.keep_original:
                    df = df.drop(columns=[col])
                
                print(f"   ✅ {col} → {new_col} (λ={lambda_val:.4f})")
            except Exception as e:
                print(f"   ❌ خطا در تبدیل {col}: {e}")
        
        # استانداردسازی ستون‌های boxcox شده
        boxcox_std_columns = [f"{col}_boxcox" for col in columns if f"{col}_boxcox" in df.columns]
        if boxcox_std_columns:
            df = self._apply_standardization(df, boxcox_std_columns)
        
        return df
    
    def _apply_standardization(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """اعمال استانداردسازی"""
        print(f"\n📏 اعمال استانداردسازی روی: {columns}")
        
        for col in columns:
            if col not in df.columns:
                print(f"   ⚠️ ستون '{col}' یافت نشد")
                continue
            
            new_col = f"{col}_std"
            
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val == 0:
                print(f"   ⚠️ {col}: انحراف معیار صفر → مقادیر به 0 تنظیم شدند")
                df[new_col] = 0.0
            else:
                df[new_col] = (df[col] - mean_val) / std_val
            
            if not self.keep_original:
                df = df.drop(columns=[col])
            
            print(f"   ✅ {col} → {new_col} (μ={mean_val:.2f}, σ={std_val:.2f})")
        
        return df