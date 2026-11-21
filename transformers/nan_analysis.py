import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaNAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر تحلیل کامل مقادیر NaN در دیتافریم
    این ترانسفورمر داده را تغییر نمی‌دهد، فقط گزارش چاپ می‌کند.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if not self.verbose:
            return df
            
        print("\n" + "=" * 60)
        print("🔍 مرحله 01_10: تحلیل کامل مقادیر NaN در دیتافریم")
        print("=" * 60)
        print(f"شکل دیتافریم: {df.shape}")
        
        # 1. اطلاعات کلی
        total_cells = df.size
        total_nan = df.isna().sum().sum()
        total_rows_with_nan = df.isna().any(axis=1).sum()
        total_columns_with_nan = df.isna().any().sum()
        
        print("\n📊 اطلاعات کلی:")
        print(f"  تعداد کل سلول‌ها: {total_cells}")
        print(f"  تعداد کل مقادیر NaN: {total_nan}")
        print(f"  درصد کل مقادیر NaN: {(total_nan/total_cells)*100:.2f}%")
        print(f"  تعداد ردیف‌های دارای حداقل یک NaN: {total_rows_with_nan}")
        print(f"  تعداد ستون‌های دارای حداقل یک NaN: {total_columns_with_nan}")
        
        # 2. تحلیل ستون‌ها
        nan_per_column = df.isna().sum()
        columns_with_nan = nan_per_column[nan_per_column > 0]
        
        if not columns_with_nan.empty:
            print("\n📉 ستون‌های دارای NaN:")
            for col, count in columns_with_nan.items():
                percentage = (count / len(df)) * 100
                print(f"  • {col}: {count} NaN ({percentage:.2f}%)")
            
            max_nan_col = nan_per_column.idxmax()
            max_nan_count = nan_per_column.max()
            print(f"\n⚠️ ستون با بیشترین NaN: '{max_nan_col}' با {max_nan_count} مقدار NaN")
        else:
            print("\n✅ هیچ ستونی دارای NaN نیست")
            
        # 3. خلاصه نهایی و پیشنهادات
        print("\n" + "=" * 40)
        print("💡 خلاصه و پیشنهادات:")
        print("=" * 40)
        if total_nan == 0:
            print("✅ دیتافریم هیچ مقدار NaN ندارد!")
        else:
            print(f"⚠️ دیتافریم دارای {total_nan} مقدار NaN است")
            if total_nan / total_cells > 0.3:
                print("  - درصد بالای NaN ممکن است نیاز به بررسی منبع داده داشته باشد")
            if max_nan_count / len(df) > 0.5:
                print(f"  - ستون '{max_nan_col}' دارای بیش از 50% NaN است، حذف این ستون را می‌توانید در نظر بگیرید.")
            print("  - در مرحله بعد، ردیف‌های دارای NaN حذف خواهند شد.")
            
        return df