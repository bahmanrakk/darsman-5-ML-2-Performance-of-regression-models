import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class CarAgeProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر محاسبه سن خودرو و حذف ستون year
    شامل:
    - محاسبه سن خودرو بر اساس سال جاری و ستون year
    - ایجاد ستون جدید 'age'
    - حذف ستون 'year'
    """
    
    def __init__(self):
        # دریافت سال جاری از سیستم در لحظه مقداردهی اولیه
        self.current_year = datetime.now().year
    
    def fit(self, X, y=None):
        """هیچ پارامتری برای یادگیری وجود ندارد"""
        return self
    
    def transform(self, X, y=None):
        """
        اجرای تبدیل روی داده‌ها
        """
        X = X.copy()
        
        # بررسی وجود ستون year برای جلوگیری از خطا در صورت اجرای مکرر
        if 'year' not in X.columns:
            print("\n⚠️ هشدار: ستون 'year' در داده‌ها وجود ندارد. این مرحله رد می‌شود.")
            return X
        
        print("\n" + "=" * 60)
        print("شروع محاسبه سن خودرو و حذف ستون year")
        print("=" * 60)
        
        # محاسبه سن خودروها: سال جاری - سال تولید + 1
        X['age'] = self.current_year - X['year'] + 1
        
        # نمایش اطلاعات اولیه
        print(f"📅 اطلاعات ستون year:")
        print(f"   کمترین سال: {X['year'].min()}")
        print(f"   بیشترین سال: {X['year'].max()}")
        print(f"   سال جاری: {self.current_year}")
        
        print(f"\n📊 اطلاعات ستون age (سن خودروها):")
        print(f"   کمترین سن: {X['age'].min()} سال")
        print(f"   بیشترین سن: {X['age'].max()} سال")
        print(f"   میانگین سن: {X['age'].mean():.1f} سال")
        
        # اطلاعات تکمیلی
        print(f"\n📈 اطلاعات تکمیلی:")
        print(f"   تعداد کل ردیف‌ها: {len(X):,}")
        print(f"   تعداد مقادیر NaN در year: {X['year'].isna().sum()}")
        print(f"   تعداد مقادیر NaN در age: {X['age'].isna().sum()}")
        
        # نمایش توزیع سن‌ها
        print(f"\n📋 توزیع سن‌های خودروها:")
        age_counts = X['age'].value_counts().sort_index()
        print("کمترین سن‌ها:")
        print(age_counts.head())
        print("\nبیشترین سن‌ها:")
        print(age_counts.tail())
        
        # حذف ستون year
        X = X.drop('year', axis=1)
        
        print(f"\n✅ ستون year حذف شد و ستون age اضافه گردید.")
        
        return X