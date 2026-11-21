import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر عمومی برای حذف یک یا چند ستون از دیتافریم
    """
    
    def __init__(self, columns_to_drop):
        """
        پارامترها:
            columns_to_drop: نام ستون یا لیستی از نام ستون‌هایی که باید حذف شوند
        """
        self.columns_to_drop = columns_to_drop if isinstance(columns_to_drop, list) else [columns_to_drop]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        print(f"\n🗑️ مرحله حذف ستون‌ها: {self.columns_to_drop}")
        print("-" * 40)
        
        # errors='ignore' باعث می‌شود اگر ستونی از قبل حذف شده بود، برنامه کرش نکند
        cols_to_actually_drop = [col for col in self.columns_to_drop if col in df.columns]
        
        if cols_to_actually_drop:
            df = df.drop(columns=cols_to_actually_drop)
            print(f"✅ ستون‌های {cols_to_actually_drop} با موفقیت حذف شدند.")
        else:
            print(f"⚠️ هیچ‌کدام از ستون‌های {self.columns_to_drop} در داده‌ها یافت نشدند.")
            
        print(f"📊 ابعاد جدید دیتافریم: {df.shape}")
        
        return df