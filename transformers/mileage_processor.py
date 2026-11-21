import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class MileageProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر پردازش و استانداردسازی ستون mileage
    شامل:
    - حذف فضاهای اضافی
    - استانداردسازی مقادیر گمشده
    - تبدیل واحدهای مختلف (km/kg و kmpl) به یک واحد استاندارد (kmpl)
    - گرد کردن نتایج به 4 رقم اعشار
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
        
        پارامترها:
            X: دیتافریم ورودی
            y: نادیده گرفته می‌شود
        
        بازگشت:
            pd.DataFrame: دیتافریم با ستون mileage پردازش شده
        """
        X = X.copy()
        
        print("\n" + "=" * 60)
        print("شروع پردازش و استانداردسازی ستون Mileage")
        print("=" * 60)
        
        # مرحله 1: حذف فضاهای اضافی
        print("\n📝 مرحله 1: حذف فضاهای اضافی")
        print("-" * 40)
        X = self._strip_whitespace(X)
        
        # مرحله 2: استانداردسازی مقادیر گمشده
        print("\n🔄 مرحله 2: استانداردسازی مقادیر گمشده")
        print("-" * 40)
        X = self._standardize_missing_values(X)
        
        # مرحله 3: بررسی مقادیر NaN در fuel
        print("\n🔍 مرحله 3: بررسی مقادیر NaN در fuel")
        print("-" * 40)
        self._check_nan_fuel(X)
        
        # مرحله 4: پردازش و تبدیل واحدها
        print("\n⚙️ مرحله 4: پردازش و تبدیل واحدها")
        print("-" * 40)
        X = self._process_mileage_values(X)
        
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
        df['fuel'] = df['fuel'].astype(str).str.strip()
        df['mileage'] = df['mileage'].astype(str).str.strip()
        print("✅ فضاهای اضافی حذف شدند")
        return df
    
    def _standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """تبدیل مقادیر خالی و رشته‌های معادل NaN به np.nan"""
        missing_values = ['', 'nan', 'NaN', 'None']
        df['fuel'] = df['fuel'].replace(missing_values, np.nan)
        df['mileage'] = df['mileage'].replace(missing_values, np.nan)
        print("✅ مقادیر گمشده استانداردسازی شدند")
        return df
    
    def _check_nan_fuel(self, df: pd.DataFrame) -> None:
        """بررسی و نمایش ردیف‌هایی که fuel برابر NaN است"""
        nan_fuel_mask = df['fuel'].isna()
        if nan_fuel_mask.any():
            print(f"⚠️ تعداد ردیف‌های با fuel NaN: {nan_fuel_mask.sum()}")
            nan_fuel_rows = df[nan_fuel_mask]
            for idx, row in nan_fuel_rows.head(5).iterrows():
                print(f"  ردیف {idx}: mileage='{row['mileage']}'")
            if len(nan_fuel_rows) > 5:
                print(f"  ... و {len(nan_fuel_rows) - 5} ردیف دیگر")
        else:
            print("✅ هیچ ردیفی با fuel NaN یافت نشد")
    
    def _process_mileage_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """پردازش و تبدیل مقادیر mileage"""
        # فقط ردیف‌هایی که هر دو مقدار معتبر دارند
        valid_mask = df['fuel'].notna() & df['mileage'].notna()
        
        # اعمال تابع پردازش روی ردیف‌های معتبر
        df.loc[valid_mask, 'mileage'] = df[valid_mask].apply(
            self._process_single_row, axis=1
        )
        
        print(f"✅ تعداد ردیف‌های پردازش شده: {valid_mask.sum()}")
        return df
    
    def _process_single_row(self, row: pd.Series):
        """
        پردازش یک ردیف و محاسبه mileage نهایی
        
        پارامترها:
            row: یک ردیف از دیتافریم
        
        بازگشت:
            float یا np.nan: مقدار mileage پردازش شده
        """
        try:
            mileage_str = str(row['mileage'])
            fuel_type = str(row['fuel'])
            
            # اگر مقدار اصلی NaN بود
            if pd.isna(row['mileage']) or mileage_str.lower() in ['nan', 'none', '']:
                return np.nan
            
            # الگوی regex برای پیدا کردن عدد و واحد
            pattern = r'(\d+\.?\d*)\s*(kmpl|km/kg)'
            match = re.search(pattern, mileage_str, re.IGNORECASE)
            
            if match:
                number_str = match.group(1)
                unit = match.group(2).lower()
                
                # بررسی معتبر بودن عدد
                if not self._is_valid_number(number_str):
                    print(f"⚠️ هشدار: عدد نامعتبر '{number_str}' در ردیف {row.name}")
                    return np.nan
                
                try:
                    number = float(number_str)
                except ValueError:
                    print(f"⚠️ هشدار: تبدیل ناموفق برای '{number_str}' در ردیف {row.name}")
                    return np.nan
                
                if unit == 'kmpl':
                    return round(number, self.decimal_places)
                
                elif unit == 'km/kg':
                    return self._convert_kg_to_kmpl(number, fuel_type, row.name)
                
                else:
                    return np.nan
            else:
                # اگر الگو تطبیق نکرد، فقط عدد را استخراج کن
                return self._extract_number_only(mileage_str, row.name)
        
        except Exception as e:
            print(f"⚠️ خطا در پردازش ردیف {row.name}: {e}")
            return np.nan
    
    def _convert_kg_to_kmpl(self, number: float, fuel_type: str, row_idx) -> float:
        """
        تبدیل km/kg به kmpl بر اساس نوع سوخت
        
        پارامترها:
            number: مقدار عددی mileage
            fuel_type: نوع سوخت
            row_idx: ایندکس ردیف برای گزارش خطا
        
        بازگشت:
            float: مقدار تبدیل شده یا np.nan
        """
        conversion_factors = {
            'diesel': 0.832,
            'petrol': 0.745,
            'cng': 0.17,
            'lpg': 0.54
        }
        
        fuel_lower = fuel_type.lower()
        if fuel_lower in conversion_factors:
            result = number * conversion_factors[fuel_lower]
            return round(result, self.decimal_places)
        else:
            print(f"⚠️ هشدار: نوع سوخت نامشخص '{fuel_type}' در ردیف {row_idx}")
            return np.nan
    
    def _extract_number_only(self, mileage_str: str, row_idx):
        """استخراج عدد از رشته بدون واحد"""
        number_pattern = r'(\d+\.?\d*)'
        number_match = re.search(number_pattern, mileage_str)
        
        if number_match:
            number_str = number_match.group(1)
            
            if not self._is_valid_number(number_str):
                print(f"⚠️ هشدار: عدد نامعتبر '{number_str}' در ردیف {row_idx}")
                return np.nan
            
            try:
                number = float(number_str)
                return round(number, self.decimal_places)
            except ValueError:
                print(f"⚠️ هشدار: تبدیل ناموفق برای '{number_str}' در ردیف {row_idx}")
                return np.nan
        else:
            return np.nan
    
    @staticmethod
    def _is_valid_number(number_str: str) -> bool:
        """
        بررسی معتبر بودن یک رشته عددی
        فقط یک نقطه مجاز است
        
        پارامترها:
            number_str: رشته عددی
        
        بازگشت:
            bool: True اگر معتبر باشد
        """
        valid_number_pattern = r'^\d+\.?\d*$|^\d*\.?\d+$'
        return bool(re.match(valid_number_pattern, number_str))
    
    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """اطمینان از عددی بودن ستون mileage"""
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        print("✅ ستون mileage به نوع عددی تبدیل شد")
        return df
    
    def _print_summary(self, df: pd.DataFrame) -> None:
        """نمایش خلاصه پردازش و آمار"""
        print(f"\n📈 خلاصه پردازش:")
        print(f"  تعداد کل ردیف‌ها: {len(df)}")
        print(f"  تعداد مقادیر NaN در mileage: {df['mileage'].isna().sum()}")
        
        # آمار مقادیر معتبر
        valid_mileage = df['mileage'].dropna()
        if len(valid_mileage) > 0:
            print(f"\n📊 آمار مقادیر معتبر mileage:")
            print(f"  تعداد: {len(valid_mileage)}")
            print(f"  میانگین: {valid_mileage.mean():.4f}")
            print(f"  انحراف معیار: {valid_mileage.std():.4f}")
            print(f"  حداقل: {valid_mileage.min():.4f}")
            print(f"  حداکثر: {valid_mileage.max():.4f}")
            print(f"  میانه: {valid_mileage.median():.4f}")
        else:
            print("\n⚠️ هیچ مقدار عددی معتبری وجود ندارد")
        
        # نمایش نمونه
        print(f"\n🔍 نمونه‌هایی از نتایج (10 ردیف اول):")
        print("*" * 50)
        sample = df[['fuel', 'mileage']].head(10)
        for idx, row in sample.iterrows():
            if pd.isna(row['mileage']):
                print(f"  ردیف {idx}: fuel='{row['fuel']}', mileage=NaN")
            else:
                print(f"  ردیف {idx}: fuel='{row['fuel']}', mileage={row['mileage']:.4f}")