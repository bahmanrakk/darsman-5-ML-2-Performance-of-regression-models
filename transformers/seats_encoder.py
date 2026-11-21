import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SeatsEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str = 'seats'):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.column_name not in df.columns:
            return df

        print("\n" + "=" * 60)
        print("💺 مرحله کدگذاری و کلامپ ستون Seats")
        print("=" * 60)

        conditions = [
            (df[self.column_name] <= 4),
            (df[self.column_name] == 5),
            (df[self.column_name].isin([6, 7])),
            (df[self.column_name] >= 8)
        ]
        choices = [1, 2, 3, 4]
        
        df['seats_category'] = np.select(conditions, choices, default=2)
        df[self.column_name] = df[self.column_name].clip(lower=4, upper=8)
        
        print(f"✅ ستون '{self.column_name}' در بازه [4, 8] کلامپ شد.")
        print(f"✅ ستون جدید 'seats_category' با 4 دسته ایجاد شد.")
        
        return df