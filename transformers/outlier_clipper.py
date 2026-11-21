import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clipping_configs: list):
        """
        clipping_configs: لیستی از دیکشنری‌ها شامل 'column', 'min_thresh', 'max_thresh'
        """
        self.clipping_configs = clipping_configs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        print("\n" + "=" * 60)
        print("✂️ مرحله کلامپ کردن (Clipping) داده‌های پرت")
        print("=" * 60)
        
        for config in self.clipping_configs:
            col = config['column']
            min_t = config['min_thresh']
            max_t = config['max_thresh']
            
            if col in df.columns:
                before_min = df[col].min()
                before_max = df[col].max()
                
                # اعمال کلامپ
                df[col] = df[col].clip(lower=min_t, upper=max_t)
                
                clipped_lower = (df[col] == min_t).sum() if before_min < min_t else 0
                clipped_upper = (df[col] == max_t).sum() if before_max > max_t else 0
                
                print(f"✅ ستون '{col}':")
                print(f"   محدوده: [{min_t}, {max_t}]")
                print(f"   مقادیر اصلاح شده: {clipped_lower} (پایین), {clipped_upper} (بالا)")
            else:
                print(f"⚠️ هشدار: ستون '{col}' در داده‌ها یافت نشد و از لیست حذف شد.")
                
        return df