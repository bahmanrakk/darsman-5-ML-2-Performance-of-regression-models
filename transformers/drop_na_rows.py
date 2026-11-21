import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropNaRowsTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر حذف تمام ردیف‌هایی که دارای حداقل یک مقدار NaN هستند
    """
    
    def __init__(self, reset_index: bool = True, verbose: bool = True):
        self.reset_index = reset_index
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if self.verbose:
            original_shape = df.shape
            original_rows_with_nan = df.isna().any(axis=1).sum()
            print("\n" + "=" * 60)
            print("🗑️ مرحله 01_11: حذف ردیف‌های دارای مقادیر گم‌شده (NaN)")
            print("=" * 60)
            print(f"📊 اطلاعات قبل از حذف:")
            print(f"   شکل دیتافریم: {original_shape}")
            print(f"   تعداد ردیف‌های دارای NaN: {original_rows_with_nan}")
            print(f"   درصد ردیف‌های دارای NaN: {(original_rows_with_nan/original_shape[0])*100:.2f}%")
        
        # حذف ردیف‌های دارای NaN
        result_df = df.dropna()
        
        if self.reset_index:
            result_df = result_df.reset_index(drop=True)
            
        if self.verbose:
            new_shape = result_df.shape
            rows_removed = original_shape[0] - new_shape[0]
            print(f"\n✅ عملیات حذف انجام شد:")
            print(f"   شکل جدید دیتافریم: {new_shape}")
            print(f"   تعداد ردیف‌های حذف شده: {rows_removed}")
            print(f"   تعداد ردیف‌های باقی‌مانده: {new_shape[0]}")
            if rows_removed > 0:
                print(f"   درصد داده‌های باقی‌مانده: {(new_shape[0]/original_shape[0])*100:.2f}%")
            else:
                print("   هیچ ردیفی حذف نشد - دیتافریم فاقد NaN بود")
                
        return result_df