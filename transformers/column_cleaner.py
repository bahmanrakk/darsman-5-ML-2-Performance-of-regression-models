import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CleanColumnNamesTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر پاک‌سازی نام ستون‌ها
    - حذف فاصله‌های اضافی
    - جایگزینی فاصله‌های میانی با _
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        print("\n  ↳ پاک‌سازی نام ستون‌ها")
        X.columns = (
            X.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
        )
        print("⚙️ نام ستون‌ها پس از پاک‌سازی:")
        for col in X.columns.tolist():
            print(f"    • {col}")
        
        return X