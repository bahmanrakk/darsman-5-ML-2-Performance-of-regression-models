import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر رمزگذاری ویژگی‌های دسته‌ای
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        print("\n" + "=" * 60)
        print("شروع تبدیل ویژگی‌های دسته‌ای (Categorical Encoding)")
        print("=" * 60)
        
        # 1. One-Hot Encoding برای fuel و seller_type
        print("انجام One-Hot Encoding برای fuel و seller_type...")
        df = pd.get_dummies(df, columns=['fuel', 'seller_type'], drop_first=False)
        
        # ✨ اصلاح: جایگزینی فاصله‌ها با آندرلاین در نام تمام ستون‌ها
        # این کار باعث می‌شود ستون‌های جدیدی که توسط get_dummies ساخته شده‌اند
        # نیز مانند سایر ستون‌ها فرمت snake_case داشته باشند (مثلاً seller_type_Trustmark_Dealer)
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
        
        # 2. تبدیل transmission به مقدار باینری
        print("تبدیل transmission به مقادیر باینری...")
        df['transmission_bin'] = df['transmission'].map({
            'Manual': 0,
            'Automatic': 1
        })
        
        # 3. تبدیل owner به مقادیر عددی ترتیبی
        owner_map = {
            'First Owner': 5,
            'Test Drive Car': 4,
            'Second Owner': 3,
            'Third Owner': 2,
            'Fourth & Above Owner': 1
        }
        print("تبدیل owner به مقادیر ترتیبی...")
        df['owner_ord'] = df['owner'].map(owner_map)
        
        # حذف ستون‌های اصلی که جایگزین شده‌اند
        cols_to_drop = ['transmission', 'owner']
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"✅ ستون‌های اصلی {existing_cols} حذف شدند.")
        
        # --- بخش گزارش‌دهی ---
        print("\n" + "=" * 50)
        print("نتایج تبدیل‌ها:")
        print("=" * 50)
        
        original_columns = ['fuel', 'seller_type', 'transmission', 'owner']
        new_columns = [col for col in df.columns if col not in original_columns]
        print(f"\nتعداد ستون‌های جدید ایجاد شده: {len(new_columns)}")
        
        if 'transmission_bin' in df.columns:
            print(f"مقادیر transmission_bin: {df['transmission_bin'].dropna().unique()}")
        if 'owner_ord' in df.columns:
            print(f"مقادیر owner_ord: {df['owner_ord'].dropna().unique()}")
            print(f"\nتوزیع مقادیر owner_ord:\n{df['owner_ord'].value_counts().sort_index(ascending=False)}")
        
        fuel_cols = [col for col in df.columns if 'fuel_' in col]
        seller_cols = [col for col in df.columns if 'seller_type_' in col]
        print(f"\nستون‌های one-hot برای fuel: {fuel_cols}")
        print(f"ستون‌های one-hot برای seller_type: {seller_cols}")
        
        print("\n" + "=" * 60)
        print("خلاصه نهایی تبدیل‌ها:")
        print(f"تعداد ستون‌های بعد از تبدیل: {len(df.columns)}")
        print("=" * 60)
        
        return df