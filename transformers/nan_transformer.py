import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class StandardizeNaNTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر استانداردسازی مقادیر گمشده
    - تبدیل مقادیر متنی رایج (NA, null, None, ...) به np.nan
    - حذف فاصله‌های ابتدایی و انتهایی رشته‌ها
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """هیچ پارامتری برای یادگیری وجود ندارد"""
        return self
    
    def transform(self, X, y=None):
        """
        اجرای تبدیل روی داده‌ها
        
        پارامترها:
            X: دیتافریم ورودی
            y: نادیده گرفته می‌شود (برای سازگاری با sklearn)
        
        بازگشت:
            pd.DataFrame: دیتافریم با مقادیر گمشده استاندارد شده
        """
        X = X.copy()
        
        print("\n🔄 استاندارسازی مقادیر گمشده و حذف فاصله‌ها")
        
        # حذف فاصله‌های ابتدای/انتهای رشته‌ها
        X = X.replace(r"^\s+|\s+$", "", regex=True)
        
        # استانداردسازی مقادیر گمشده
        nan_pattern = r'(?i)^(nan|na|n/a|null|none|missing|-|\.)?$'
        X = X.replace(nan_pattern, np.nan, regex=True)
        
        print("⚙️ فاصله‌ها حذف و مقادیر گمشده استانداردسازی شدند.")
        return X