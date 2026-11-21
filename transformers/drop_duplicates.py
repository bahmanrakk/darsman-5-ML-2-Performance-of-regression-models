import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر شناسایی و حذف ردیف‌های تکراری
    """
    
    def __init__(self, subset=None, keep: str = 'first', remove_duplicates: bool = True, verbose: bool = True):
        self.subset = subset
        self.keep = keep
        self.remove_duplicates = remove_duplicates
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔍 مرحله 01_12: شناسایی و حذف ردیف‌های تکراری")
            print("=" * 60)
            print(f"شکل دیتافریم ورودی: {df.shape}")
            
        duplicate_mask = df.duplicated(subset=self.subset, keep=self.keep)
        duplicate_count = duplicate_mask.sum()
        
        group_cols = self.subset if self.subset else df.columns.tolist()
        duplicate_groups = df.groupby(group_cols).size()
        duplicate_details = duplicate_groups[duplicate_groups > 1]
        
        if self.verbose:
            print(f"\n📊 نتایج شناسایی ردیف‌های تکراری:")
            print(f"  تعداد کل ردیف‌ها: {len(df)}")
            print(f"  تعداد ردیف‌های تکراری: {duplicate_count}")
            print(f"  درصد ردیف‌های تکراری: {(duplicate_count/len(df))*100:.2f}%")
            
            if duplicate_count > 0:
                print(f"\n🧮 تعداد گروه‌های تکراری: {len(duplicate_details)}")
                print(f"  بیشترین تکرار یک گروه: {duplicate_details.max()} بار")
                
                duplicate_rows = df[duplicate_mask]
                print(f"\n📝 نمونه‌ای از ردیف‌های تکراری ({min(3, len(duplicate_rows))} مورد اول):")
                print(duplicate_rows.head(3))
        
        # حذف ردیف‌های تکراری اگر درخواست شده باشد
        if self.remove_duplicates:
            result_df = df.drop_duplicates(subset=self.subset, keep=self.keep)
            
            if self.verbose and duplicate_count > 0:
                print("\n" + "=" * 60)
                print(f"📦 پس از حذف تکراری‌ها: {len(result_df)} ردیف باقی می‌ماند")
                print(f"💾 صرفه‌جویی: {len(df) - len(result_df)} ردیف حذف شد")
                print(f"📊 شکل نهایی دیتافریم: {result_df.shape}")
        else:
            result_df = df
            
        return result_df