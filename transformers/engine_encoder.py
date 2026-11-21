import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EngineEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str = 'engine'):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.column_name not in df.columns:
            return df

        print("\n" + "=" * 60)
        print("🔧 مرحله کدگذاری و کلامپ ستون Engine")
        print("=" * 60)

        # 1. ایجاد دسته‌بندی بر اساس داده‌های اصلی
        bins = [0, 600, 1200, 1800, 2400, 3000, float('inf')]
        labels = [1, 2, 3, 4, 5, 6]
        
        df['engine_category_encoded'] = pd.cut(
            df[self.column_name], bins=bins, labels=labels, include_lowest=True
        ).astype(int)

        # 2. کلامپ کردن ستون اصلی
        original_min, original_max = df[self.column_name].min(), df[self.column_name].max()
        df[self.column_name] = df[self.column_name].clip(lower=600, upper=3000)
        
        print(f"✅ ستون '{self.column_name}' در بازه [600, 3000] کلامپ شد.")
        print(f"✅ ستون جدید 'engine_category_encoded' با 6 دسته ایجاد شد.")
        
        return df