"""
╔══════════════════════════════════════════════════════════════╗
║  پیش‌پردازش داده‌ها برای مدل‌سازی                           ║
║  Data Preprocessing for Modeling                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.base import BaseEstimator, TransformerMixin


# ══════════════════════════════════════════════════════════════
#  Transformer 1 | تبدیل bool به numeric
# ══════════════════════════════════════════════════════════════

class BoolToNumericTransformer(BaseEstimator, TransformerMixin):
    """
    تبدیل تمام ستون‌های bool به int (0 و 1)
    
    این transformer:
    1. تمام ستون‌های bool را شناسایی می‌کند
    2. آن‌ها را به int8 تبدیل می‌کند
    3. گزارش می‌دهد که چه ستون‌هایی تبدیل شدند
    """
    
    def __init__(self):
        self.bool_columns_ = None
    
    def fit(self, X, y=None):
        """شناسایی ستون‌های bool"""
        self.bool_columns_ = X.select_dtypes(include=['bool']).columns.tolist()
        return self
    
    def transform(self, X, y=None):
        """تبدیل ستون‌های bool به int"""
        X = X.copy()
        
        print("\n" + "=" * 65)
        print("🔄  تبدیل ستون‌های bool به numeric")
        print("=" * 65)
        
        if not self.bool_columns_:
            print("   ✅ هیچ ستون bool یافت نشد.")
            return X
        
        print(f"\n   🔍 تعداد ستون‌های bool یافت‌شده: {len(self.bool_columns_)}")
        print("   📋 لیست ستون‌ها:")
        
        for col in self.bool_columns_:
            # تبدیل به int8 (کمترین مصرف حافظه)
            X[col] = X[col].astype('int8')
            print(f"      • {col}: bool → int8")
        
        print(f"\n   ✅ تمام ستون‌های bool به int8 تبدیل شدند.")
        print("=" * 65)
        
        return X


# ══════════════════════════════════════════════════════════════
#  Transformer 2 | بررسی و تبدیل category به numeric
# ══════════════════════════════════════════════════════════════

class CategoryToNumericTransformer(BaseEstimator, TransformerMixin):
    """
    تبدیل ستون‌های category به numeric (در صورت نیاز)
    
    این transformer:
    1. ستون‌های category را شناسایی می‌کند
    2. اگر مقادیر عددی باشند، به numeric تبدیل می‌کند
    3. در غیر این صورت، هشدار می‌دهد
    """
    
    def __init__(self):
        self.category_columns_ = None
    
    def fit(self, X, y=None):
        """شناسایی ستون‌های category"""
        self.category_columns_ = X.select_dtypes(include=['category']).columns.tolist()
        return self
    
    def transform(self, X, y=None):
        """تبدیل ستون‌های category به numeric"""
        X = X.copy()
        
        if not self.category_columns_:
            return X
        
        print("\n" + "=" * 65)
        print("🔄  بررسی ستون‌های category")
        print("=" * 65)
        
        converted = []
        skipped = []
        
        for col in self.category_columns_:
            try:
                # تلاش برای تبدیل به numeric
                X[col] = pd.to_numeric(X[col])
                converted.append(col)
            except (ValueError, TypeError):
                skipped.append(col)
        
        if converted:
            print(f"\n   ✅ ستون‌های تبدیل‌شده ({len(converted)}):")
            for col in converted:
                print(f"      • {col}")
        
        if skipped:
            print(f"\n   ⚠️ ستون‌های غیرقابل تبدیل ({len(skipped)}):")
            for col in skipped:
                print(f"      • {col}")
            print("      این ستون‌ها نیاز به encoding دارند.")
        
        print("=" * 65)
        
        return X


# ══════════════════════════════════════════════════════════════
#  تابع اصلی | اجرای پیش‌پردازش داده
# ══════════════════════════════════════════════════════════════

def preprocess_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    اجرای تمام مراحل پیش‌پردازش برای آماده‌سازی داده‌ها برای مدل‌سازی
    
    پارامترها:
        df: دیتافریم ورودی
    
    بازگشت:
        دیتافریم آماده برای مدل‌سازی
    """
    print("\n" + "=" * 65)
    print("🔧  شروع پیش‌پردازش داده برای مدل‌سازی")
    print("=" * 65)
    
    # نمایش نوع داده‌ها قبل از تبدیل
    print("\n📊  نوع داده‌ها قبل از تبدیل:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} ستون")
    
    # مرحله 1: تبدیل bool به numeric
    bool_transformer = BoolToNumericTransformer()
    df = bool_transformer.fit_transform(df)
    
    # مرحله 2: تبدیل category به numeric
    category_transformer = CategoryToNumericTransformer()
    df = category_transformer.fit_transform(df)
    
    # نمایش نوع داده‌ها بعد از تبدیل
    print("\n📊  نوع داده‌ها بعد از تبدیل:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} ستون")
    
    print("\n" + "=" * 65)
    print("✅  پیش‌پردازش داده برای مدل‌سازی کامل شد.")
    print("=" * 65)
    
    return df