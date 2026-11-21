import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class TorqueProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر پردازش کامل ستون torque
    شامل: پاک‌سازی، تبدیل فرمت، استخراج مقادیر، تبدیل واحد و به‌روزرسانی
    """
    
    def __init__(self, drop_torque_column: bool = True):
        """
        پارامترها:
            drop_torque_column: آیا ستون اصلی torque بعد از پردازش حذف شود؟
        """
        self.drop_torque_column = drop_torque_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        print("\n" + "=" * 60)
        print("شروع پردازش کامل داده‌های Torque")
        print("=" * 60)
        
        # مرحله 1: پاک‌سازی اولیه
        print("\n📝 مرحله 1: پاکسازی اولیه داده‌ها")
        print("-" * 40)
        X = self._remove_nan_torque(X)
        print(f"✅ ابعاد: {X.shape}")
        
        # مرحله 2: تبدیل فرمت‌ها
        print("\n🔄 مرحله 2: تبدیل فرمت‌های Torque")
        print("-" * 40)
        X = self._convert_torque_format(X)
        print(f"✅ ابعاد: {X.shape}")
        
        # مرحله 3: استخراج Nm@rpm
        print("\n🔧 مرحله 3: استخراج مقادیر از فرمت Nm@rpm")
        print("-" * 40)
        X = self._extract_nm_rpm(X)
        print(f"✅ ابعاد: {X.shape}")
        
        # مرحله 4: استخراج kgm@rpm
        print("\n⚡ مرحله 4: استخراج مقادیر از فرمت kgm@rpm")
        print("-" * 40)
        X = self._extract_kgm_rpm(X)
        print(f"✅ ابعاد: {X.shape}")
        
        # مرحله 5: به‌روزرسانی مقادیر خاص
        print("\n🎯 مرحله 5: به‌روزرسانی مقادیر خاص")
        print("-" * 40)
        torque_rpm_mapping = {
            400: 2125.0,
            480: 2125.0
            # مقادیر دیگر را اینجا اضافه کنید
        }
        X = self._update_multiple_torque_rpm_power(X, torque_rpm_mapping)
        print(f"✅ ابعاد: {X.shape}")
        
        # حذف ستون torque در صورت نیاز
        if self.drop_torque_column and 'torque' in X.columns:
            X = X.drop('torque', axis=1)
            print(f"\n🗑️ ستون 'torque' حذف شد. ابعاد نهایی: {X.shape}")
        
        # آمار نهایی
        print("\n📊 آمار نهایی پردازش:")
        print("-" * 40)
        summary_stats = X[['torque_nm', 'torque_rpm']].agg(
            ['max', 'min', 'mean', 'median']
        ).round(3)
        print(summary_stats.to_string())
        
        return X
    
    # ------------------------------------------------------------------
    # متدهای داخلی
    # ------------------------------------------------------------------
    
    def _remove_nan_torque(self, df: pd.DataFrame) -> pd.DataFrame:
        """پاک‌سازی اولیه ستون torque و ایجاد ستون‌های جدید"""
        df['torque'] = df['torque'].apply(
            lambda x: re.sub(
                r'\s+', ' ',
                str(x).replace('"', '')
                       .replace(',', '')
                       .replace(' ', '')
                       .replace('~', '-')
                       .lower()
                       .replace('at', '@')
            ).strip() if pd.notna(x) else x
        )
        
        # حذف ردیف‌هایی که رشته خالی شده‌اند
        df = df[df['torque'].str.strip() != '']
        
        # ایجاد ستون‌های جدید
        df['torque_nm'] = np.nan
        df['torque_rpm'] = np.nan
        
        return df
    
    def _convert_torque_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """تبدیل فرمت‌های مختلف به فرمت استاندارد"""
        original_torque = df['torque'].copy()
        
        # تعریف الگوها و توابع جایگزینی
        patterns = [
            # فرمت 1: عدد@عدد+/-عدد(nm@rpm)
            (r'(\d+\.?\d*)\s*@\s*(\d+)\s*[\+\-]\s*\/?\s*[\+\-]?\s*\d+\s*\(\s*nm\s*@\s*rpm\s*\)',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 2: عددnm@عدد-عددrpm → میانگین
            (r'(\d+\.?\d*)\s*nm\s*@\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*rpm',
             lambda m: f"{m.group(1)}nm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 3: عددnm@عدد+/-عددrpm
            (r'(\d+\.?\d*)\s*nm\s*@\s*(\d+)\s*\+\/-\s*(\d+)\s*rpm',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 4: عدد@عدد-عدد(kgm@rpm) → میانگین
            (r'(\d+\.?\d*)\s*@\s*(\d+)\s*-\s*(\d+)\s*\(\s*kgm\s*@\s*rpm\s*\)',
             lambda m: f"{m.group(1)}kgm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 5: عددkgm@عدد-عددrpm → میانگین
            (r'(\d+\.?\d*)\s*kgm\s*@\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*rpm',
             lambda m: f"{m.group(1)}kgm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 6: عدد@عدد(kgm@rpm)
            (r'(\d+\.?\d*)\s*@\s*(\d+)\s*\(\s*kgm\s*@\s*rpm\s*\)',
             lambda m: f"{m.group(1)}kgm@{m.group(2)}rpm"),
            
            # فرمت 7: عدد@عددrpm → nm
            (r'(\d+\.?\d*)\s*@\s*(\d+)\s*rpm',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 8: عدد@عدد-عددrpm → میانگین nm
            (r'(\d+\.?\d*)\s*@\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*rpm',
             lambda m: f"{m.group(1)}nm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 9: عددnm(عددkgm)@عددrpm
            (r'(\d+\.?\d*)\s*nm\s*\(\s*\d+\.?\d*\s*kgm\s*\)\s*@\s*(\d+\.?\d*)\s*rpm',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 10: عددnm@عدد-عدد (بدون rpm) → میانگین
            (r'(\d+\.?\d*)\s*nm\s*@\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$',
             lambda m: f"{m.group(1)}nm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 11: عددnm@عدد (بدون rpm)
            (r'(\d+\.?\d*)\s*nm\s*@\s*(\d+\.?\d*)$',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 12: عددnm/عددrpm
            (r'(\d+\.?\d*)\s*nm\s*\/\s*(\d+\.?\d*)\s*rpm',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 13: عدد@عدد-عدد (بدون واحد) → میانگین nm
            (r'(\d+\.?\d*)\s*@\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$',
             lambda m: f"{m.group(1)}nm@{self._avg(m.group(2), m.group(3))}rpm"),
            
            # فرمت 14: عدد(عدد)@عدد
            (r'(\d+\.?\d*)\s*\(\s*\d+\.?\d*\s*\)\s*@\s*(\d+\.?\d*)$',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 15: عدد/عدد
            (r'(\d+\.?\d*)\s*\/\s*(\d+\.?\d*)$',
             lambda m: f"{m.group(1)}nm@{m.group(2)}rpm"),
            
            # فرمت 16: عددnm → 999999rpm
            (r'(\d+\.?\d*)\s*nm$',
             lambda m: f"{m.group(1)}nm@999999rpm"),
        ]
        
        # اعمال الگوها
        for pattern, replacer in patterns:
            df['torque'] = df['torque'].apply(
                lambda x, p=pattern, r=replacer: (
                    re.sub(p, r, str(x), flags=re.IGNORECASE) 
                    if pd.notna(x) else x
                )
            )
        
        # گزارش تغییرات
        changed_count = (original_torque != df['torque']).sum()
        print(f"تعداد ردیف‌های تغییر کرده: {changed_count}")
        print(f"تعداد کل ردیف‌های پردازش شده: {len(df)}")
        
        return df
    
    def _extract_nm_rpm(self, df: pd.DataFrame) -> pd.DataFrame:
        """استخراج مقادیر از فرمت nm@rpm"""
        pattern = r'^\s*(\d+\.?\d*)\s*(Nm|nm|NM|nM)\s*@\s*(\d+\.?\d*)\s*rpm\s*$'
        
        matching_mask = df['torque'].str.match(pattern, na=False, case=False)
        matching_rows = df[matching_mask]
        
        def extract(torque_str):
            match = re.match(pattern, str(torque_str), re.IGNORECASE)
            if match:
                return float(match.group(1)), float(match.group(3))
            return None, None
        
        extracted = df.loc[matching_mask, 'torque'].apply(extract)
        df.loc[matching_mask, 'torque_nm'] = extracted.apply(
            lambda x: x[0] if x[0] else np.nan
        )
        df.loc[matching_mask, 'torque_rpm'] = extracted.apply(
            lambda x: x[1] if x[1] else np.nan
        )
        
        # rpm نامعتبر
        invalid_mask = df['torque_rpm'] == 999999
        df.loc[invalid_mask, 'torque_rpm'] = np.nan
        
        print(f"تعداد ردیف‌های پردازش شده: {len(matching_rows)}")
        print(f"تعداد ردیف‌های با rpm نامعتبر (999999): {invalid_mask.sum()}")
        
        return df
    
    def _extract_kgm_rpm(self, df: pd.DataFrame) -> pd.DataFrame:
        """استخراج مقادیر از فرمت kgm@rpm و تبدیل به Nm"""
        pattern = r'^\s*(\d+\.?\d*)\s*kgm@\s*(\d+\.?\d*)\s*rpm\s*$'
        
        matching_mask = df['torque'].str.match(pattern, na=False, case=False)
        matching_rows = df[matching_mask]
        
        def extract(torque_str):
            match = re.match(pattern, str(torque_str), re.IGNORECASE)
            if match:
                return float(match.group(1)), float(match.group(2))
            return None, None
        
        extracted = df.loc[matching_mask, 'torque'].apply(extract)
        
        # تبدیل kgm به Nm (ضریب 9.80665)
        df.loc[matching_mask, 'torque_nm'] = extracted.apply(
            lambda x: x[0] * 9.80665 if x[0] else np.nan
        )
        df.loc[matching_mask, 'torque_rpm'] = extracted.apply(
            lambda x: x[1] if x[1] else np.nan
        )
        
        invalid_mask = df['torque_rpm'] == 999999
        df.loc[invalid_mask, 'torque_rpm'] = np.nan
        
        print(f"تعداد ردیف‌های پردازش شده با فرمت kgm: {len(matching_rows)}")
        print(f"تعداد ردیف‌های با rpm نامعتبر: {invalid_mask.sum()}")
        
        return df
    
    def _update_multiple_torque_rpm_power(
        self, df: pd.DataFrame, mapping: dict
    ) -> pd.DataFrame:
        """به‌روزرسانی مقادیر خاص torque_rpm"""
        total_updated = 0
        
        for torque_nm_value, new_rpm in mapping.items():
            condition = (df['torque_nm'] == torque_nm_value) & (df['torque_rpm'].isna())
            target_rows = df[condition]
            
            if len(target_rows) > 0:
                df.loc[target_rows.index, 'torque_rpm'] = new_rpm
                total_updated += len(target_rows)
                print(f"  torque_nm={torque_nm_value}: {len(target_rows)} ردیف به‌روزرسانی شد")
        
        print(f"جمع کل به‌روزرسانی‌ها: {total_updated}")
        return df
    
    @staticmethod
    def _avg(a, b):
        """میانگین دو عدد - اگر صحیح باشد int برمی‌گرداند"""
        avg = (float(a) + float(b)) / 2
        return int(avg) if avg.is_integer() else avg