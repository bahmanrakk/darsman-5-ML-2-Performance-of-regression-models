import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class MaxPowerProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر پردازش و استانداردسازی ستون max_power
    شامل:
    - حذف فضاهای اضافی
    - استانداردسازی مقادیر گمشده
    - استخراج عدد و واحد (bhp) از رشته
    - گرد کردن نتایج به تعداد اعشار مشخص
    """
    
    def __init__(self, decimal_places: int = 4):
        """
        پارامترها:
            decimal_places: تعداد ارقام اعشار برای گرد کردن (پیش‌فرض: 4)
        """
        self.decimal_places = decimal_places
    
    def fit(self, X, y=None):
        """هیچ پارامتری برای یادگیری وجود ندارد"""
        return self
    
    def transform(self, X, y=None):
        """
        اجرای تبدیل روی داده‌ها
        """
        X = X.copy()
        
        # بررسی وجود ستون max_power برای جلوگیری از خطا
        if 'max_power' not in X.columns:
            print("⚠️ هشدار: ستون 'max_power' در داده‌ها وجود ندارد. این مرحله رد می‌شود.")
            return X
        
        print("\n" + "=" * 60)
        print("شروع پردازش و استانداردسازی ستون Max Power")
        print("=" * 60)
        
        # مرحله 1: حذف فضاهای اضافی
        print("\n📝 مرحله 1: حذف فضاهای اضافی")
        print("-" * 40)
        X = self._strip_whitespace(X)
        
        # مرحله 2: استانداردسازی مقادیر گمشده
        print("\n🔄 مرحله 2: استانداردسازی مقادیر گمشده")
        print("-" * 40)
        X = self._standardize_missing_values(X)
        
        # مرحله 3: بررسی مقادیر NaN
        print("\n🔍 مرحله 3: بررسی مقادیر NaN در max_power")
        print("-" * 40)
        self._check_nan_max_power(X)
        
        # مرحله 4: پردازش و استخراج مقادیر
        print("\n⚙️ مرحله 4: پردازش و استخراج مقادیر عددی")
        print("-" * 40)
        X = self._process_max_power_values(X)
        
        # مرحله 5: اطمینان از عددی بودن
        print("\n🔢 مرحله 5: تبدیل به نوع عددی")
        print("-" * 40)
        X = self._ensure_numeric(X)
        
        # مرحله 6: نمایش آمار نهایی
        print("\n📊 مرحله 6: آمار نهایی")
        print("-" * 40)
        self._print_summary(X)
        
        return X
    
    # ------------------------------------------------------------------
    # متدهای داخلی
    # ------------------------------------------------------------------
    
    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """حذف فضاهای ابتدایی و انتهایی رشته‌ها"""
        df['max_power'] = df['max_power'].astype(str).str.strip()
        print("✅ فضاهای اضافی حذف شدند")
        return df
    
    def _standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """تبدیل مقادیر خالی و رشته‌های معادل NaN به np.nan"""
        missing_values = ['', 'nan', 'NaN', 'None']
        df['max_power'] = df['max_power'].replace(missing_values, np.nan)
        print("✅ مقادیر گمشده استانداردسازی شدند")
        return df
    
    def _check_nan_max_power(self, df: pd.DataFrame) -> None:
        """بررسی و نمایش ردیف‌هایی که max_power برابر NaN است"""
        nan_mask = df['max_power'].isna()
        if nan_mask.any():
            print(f"⚠️ تعداد ردیف‌های با max_power NaN: {nan_mask.sum()}")
            nan_rows = df[nan_mask]
            for idx, row in nan_rows.head(5).iterrows():
                print(f"  ردیف {idx}: max_power='{row['max_power']}'")
            if len(nan_rows) > 5:
                print(f"  ... و {len(nan_rows) - 5} ردیف دیگر")
        else:
            print("✅ هیچ ردیفی با max_power NaN یافت نشد")
    
    def _process_max_power_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """پردازش و استخراج مقادیر max_power"""
        valid_mask = df['max_power'].notna()
        
        # اعمال تابع پردازش فقط روی ردیف‌های معتبر
        df.loc[valid_mask, 'max_power'] = df.loc[valid_mask, 'max_power'].apply(
            self._process_single_value
        )
        
        print(f"✅ تعداد ردیف‌های پردازش شده: {valid_mask.sum()}")
        return df
    
    def _process_single_value(self, value):
        """
        پردازش یک مقدار در ستون max_power
        """
        try:
            max_power_str = str(value)
            
            if pd.isna(value) or max_power_str.lower() in ['nan', 'none', '']:
                return np.nan
            
            # الگوی regex برای پیدا کردن عدد و واحد (bhp)
            pattern = r'(\d+\.?\d*)\s*(bhp)'
            match = re.search(pattern, max_power_str, re.IGNORECASE)
            
            if match:
                number_str = match.group(1)
                unit = match.group(2).lower()
                
                # بررسی معتبر بودن عدد
                if not self._is_valid_number(number_str):
                    return np.nan
                
                try:
                    number = float(number_str)
                except ValueError:
                    return np.nan
                
                if unit == 'bhp':
                    return round(number, self.decimal_places)
                else:
                    return np.nan
            else:
                # اگر الگو تطبیق نکرد (مثلاً واحد bhp نداشت یا عدد نبود)
                return np.nan
                
        except Exception:
            return np.nan
    
    @staticmethod
    def _is_valid_number(number_str: str) -> bool:
        """بررسی معتبر بودن یک رشته عددی (فقط یک نقطه مجاز است)"""
        valid_number_pattern = r'^\d+\.?\d*$|^\d*\.?\d+$'
        return bool(re.match(valid_number_pattern, number_str))
    
    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """اطمینان از عددی بودن ستون max_power"""
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
        print("✅ ستون max_power به نوع عددی تبدیل شد")
        return df
    
    def _print_summary(self, df: pd.DataFrame) -> None:
        """نمایش خلاصه پردازش و آمار"""
        print(f"\n📈 خلاصه پردازش:")
        print(f"  تعداد کل ردیف‌ها: {len(df)}")
        print(f"  تعداد مقادیر NaN در max_power: {df['max_power'].isna().sum()}")
        
        valid_values = df['max_power'].dropna()
        if len(valid_values) > 0:
            print(f"\n📊 آمار مقادیر معتبر max_power:")
            print(f"  تعداد: {len(valid_values)}")
            print(f"  میانگین: {valid_values.mean():.4f}")
            print(f"  انحراف معیار: {valid_values.std():.4f}")
            print(f"  حداقل: {valid_values.min():.4f}")
            print(f"  حداکثر: {valid_values.max():.4f}")
            print(f"  میانه: {valid_values.median():.4f}")
        else:
            print("\n⚠️ هیچ مقدار عددی معتبری وجود ندارد")
        
        print(f"\n🔍 نمونه‌هایی از نتایج (10 ردیف اول):")
        print("*" * 50)
        sample = df[['max_power']].head(10)
        for idx, row in sample.iterrows():
            if pd.isna(row['max_power']):
                print(f"  ردیف {idx}: max_power=NaN")
            else:
                print(f"  ردیف {idx}: max_power={row['max_power']:.4f}")