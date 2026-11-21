import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AgeEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str = 'age', min_thresh: int = 5, max_thresh: int = 25):
        self.column_name = column_name
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.column_name not in df.columns:
            return df

        print("\n" + "=" * 60)
        print("📅 مرحله کدگذاری و کلامپ ستون Age")
        print("=" * 60)

        bins = [0, 5, 10, 15, 20, float('inf')]
        labels = [1, 2, 3, 4, 5]
        
        df['age_category'] = pd.cut(
            df[self.column_name], bins=bins, labels=labels, include_lowest=True
        ).astype(int)

        df[self.column_name] = df[self.column_name].clip(lower=self.min_thresh, upper=self.max_thresh)
        
        print(f"✅ ستون '{self.column_name}' در بازه [{self.min_thresh}, {self.max_thresh}] کلامپ شد.")
        print(f"✅ ستون جدید 'age_category' با 5 دسته ایجاد شد.")
        
        return df