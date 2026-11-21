import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataInspectorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر بررسی و نمایش آمار اولیه داده‌ها
    این ترانسفورمر داده را تغییر نمی‌دهد، فقط گزارش چاپ می‌کند
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # 2.3 نمایش شکل
        print(f"\n 🔢 شکل دیتافریم: {X.shape[0]} سطر x {X.shape[1]} ستون")
        
        # 2.4 اطلاعات کلی
        print("\n--- A1: اطلاعات کلی دیتافریم ---")
        X.info()
        
        # 2.5 آمار توصیفی عددی
        print("\n--- B1: آمار توصیفی (متغیرهای عددی) ---")
        print(X.describe())
        
        # 2.6 آمار متغیرهای دسته‌ای
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            print("\n--- C1: آمار متغیرهای دسته‌ای ---")
            print(X[categorical_columns].describe().T)
        else:
            print("\n--- هیچ متغیر دسته‌ای در داده‌ها وجود ندارد ---")
        
        # 2.7 پیش‌نمایش
        print("\n--- D1: پیش‌نمایش (5 سطر اول) ---")
        print(X.head(5))
        
        # 2.8 مقادیر گمشده
        print("\n--- E1: تعداد مقادیر گمشده ---")
        missing = X.isna().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("هیچ مقدار گمشده‌ای یافت نشد.")
        
        # 2.9 گروه‌بندی ستون‌ها
        print("\n--- F1: گروه بندی ستون ها ---")
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"\nمتغیرهای عددی ({len(num_features)}):")
        print(f"  {num_features}")
        print(f"\nمتغیرهای غیرعددی ({len(cat_features)}):")
        print(f"  {cat_features}")
        
        return X