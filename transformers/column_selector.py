import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر انتخاب ستون‌های نهایی برای مدل‌سازی
    """
    
    def __init__(self, columns_to_keep: list):
        """
        پارامترها:
            columns_to_keep: لیست نام ستون‌هایی که باید حفظ شوند
        """
        self.columns_to_keep = columns_to_keep
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        print("\n" + "=" * 60)
        print("🎯 مرحله انتخاب ستون‌های نهایی")
        print("=" * 60)
        
        # بررسی وجود ستون‌ها
        missing_cols = [col for col in self.columns_to_keep if col not in df.columns]
        if missing_cols:
            print(f"⚠️ هشدار: ستون‌های زیر یافت نشدند و نادیده گرفته می‌شوند:")
            for col in missing_cols:
                print(f"   - {col}")
        
        # فیلتر کردن ستون‌های موجود
        available_cols = [col for col in self.columns_to_keep if col in df.columns]
        
        if not available_cols:
            raise ValueError("هیچ‌کدام از ستون‌های مورد نظر در دیتافریم یافت نشدند")
        
        df = df[available_cols]
        
        print(f"✅ {len(available_cols)} ستون انتخاب شدند")
        print(f"📊 ابعاد نهایی دیتافریم: {df.shape}")
        
        return df