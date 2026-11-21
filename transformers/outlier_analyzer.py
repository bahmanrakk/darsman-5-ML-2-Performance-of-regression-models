import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

# اگر ماژول persian_text را دارید، آن را ایمپورت می‌کند. در غیر این صورت، یک تابع جایگزین ساده استفاده می‌شود.
try:
    from x_persian_text import persian_text
except ImportError:
    def persian_text(text): return text


class OutlierAnalyzerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_analyze: list, generate_reports: bool = False):
        self.columns_to_analyze = columns_to_analyze
        self.generate_reports = generate_reports  # ✅ نام صحیح متغیر

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        # ✅ اصلاح خطای تایپی: استفاده از generate_reports به جای generate_cols
        if not self.generate_reports:
            print("\n⏭️ مرحله تحلیل داده‌های پرت (Outlier Analysis) رد شد (generate_reports=False)")
            return df
            
        print("\n" + "=" * 60)
        print("📊 مرحله تحلیل داده‌های پرت (Outlier Analysis)")
        print("=" * 60)
        
        valid_cols = [col for col in self.columns_to_analyze if col in df.columns]
        numeric_cols = df[valid_cols].select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            print("❌ هیچ ستون عددی معتبری برای تحلیل یافت نشد.")
            return df

        project_path = Path(__file__).parent.parent
        output_dir = project_path / 'data'
        os.makedirs(output_dir, exist_ok=True)

        for col in numeric_cols:
            print(f"\n--- تحلیل ستون '{col}' ---")
            # 1. چاپ آمار
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            outliers_count = len(df[(df[col] < lower_fence) | (df[col] > upper_fence)])
            
            print(f"  - محدوده مجاز (IQR): [{lower_fence:.2f}, {upper_fence:.2f}]")
            print(f"  - تعداد داده‌های پرت: {outliers_count}")

            # 2. اگر گزارش‌گیری فعال باشد، نمودارها را می‌سازد
            if self.generate_reports:
                self._generate_plots(df, col, output_dir)

        return df

    def _generate_plots(self, df, col, output_dir):
        # خلاصه‌ای از کد رسم نمودار برای جلوگیری از شلوغی بیش از حد
        # (شما می‌توانید کد کامل رسم نمودار 4x2 خود را اینجا جایگذاری کنید)
        col_dir = output_dir / col
        col_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(y=df[col], ax=axes[0], color="skyblue")
        axes[0].set_title(persian_text(f'Box Plot: {col}'))
        
        axes[1].hist(df[col].dropna(), bins=30, color='green', alpha=0.7)
        axes[1].set_title(persian_text(f'Histogram: {col}'))
        
        plt.tight_layout()
        save_path = col_dir / f"outlier_analysis_{col}.png"
        fig.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ✔️ نمودار ذخیره شد: {save_path}")