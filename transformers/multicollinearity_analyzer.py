import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin


class MulticollinearityAnalyzerTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر تحلیل چندهمخطی‌بودگی (Multicollinearity)
    این ترانسفورمر فقط گزارش تولید می‌کند و داده را تغییر نمی‌دهد
    """
    
    def __init__(self, columns_to_analyze: list, corr_threshold: float = 0.75, 
                 vif_threshold: float = 5.0, verbose: bool = True):
        """
        پارامترها:
            columns_to_analyze: لیست ستون‌هایی که باید تحلیل شوند
            corr_threshold: آستانه همبستگی برای گزارش
            vif_threshold: آستانه VIF برای گزارش
            verbose: آیا گزارش چاپ شود؟
        """
        self.columns_to_analyze = columns_to_analyze
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if not self.verbose:
            return df
        
        print("\n" + "=" * 60)
        print("📊 مرحله تحلیل چندهمخطی‌بودگی (Multicollinearity)")
        print("=" * 60)
        
        # بررسی وجود ستون‌ها
        available_columns = [col for col in self.columns_to_analyze if col in df.columns]
        missing_columns = [col for col in self.columns_to_analyze if col not in df.columns]
        
        if missing_columns:
            print(f"\n⚠️ ستون‌های ناموجود: {missing_columns}")
        
        if not available_columns:
            print("\n❌ هیچ یک از ستون‌های مورد نظر در دیتافریم وجود ندارد.")
            return df
        
        print(f"\n✅ ستون‌های مورد تحلیل: {len(available_columns)} ستون")
        
        # تبدیل ستون‌ها به عددی
        temp_df = df[available_columns].copy()
        for col in available_columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        
        # حذف ردیف‌های دارای NaN
        temp_df = temp_df.dropna()
        
        if len(temp_df) < 2:
            print("❌ تعداد نمونه‌ها برای تحلیل کافی نیست.")
            return df
        
        # محاسبه ماتریس همبستگی
        corr_matrix = temp_df.corr()
        
        # محاسبه VIF
        try:
            X_array = temp_df.values.astype(float)
            X_const = sm.add_constant(X_array)
            
            vif_values = []
            for i in range(1, X_const.shape[1]):
                try:
                    vif = variance_inflation_factor(X_const, i)
                    vif_values.append(vif)
                except Exception as e:
                    print(f"⚠️ خطا در محاسبه VIF برای ستون {i}: {e}")
                    vif_values.append(np.inf)
            
            vif_data = pd.DataFrame({
                'feature': available_columns,
                'VIF': vif_values
            })
        except Exception as e:
            print(f"❌ خطا در محاسبه VIF: {e}")
            vif_data = pd.DataFrame({'feature': available_columns, 'VIF': [np.nan] * len(available_columns)})
        
        # شناسایی زوج‌های با همبستگی بالا
        high_corr_pairs = []
        abs_corr = corr_matrix.abs()
        for i in range(len(available_columns)):
            for j in range(i + 1, len(available_columns)):
                if abs_corr.iloc[i, j] > self.corr_threshold:
                    high_corr_pairs.append((available_columns[i], available_columns[j], abs_corr.iloc[i, j]))
        
        # شناسایی ستون‌های با VIF بالا
        high_vif_df = vif_data[vif_data['VIF'] > self.vif_threshold]
        
        # چاپ گزارش
        print("\n📋 ماتریس همبستگی:")
        print(corr_matrix.round(2))
        
        print("\n🔍 جدول VIF:")
        print(vif_data.round(2))
        
        print("\n💡 پیشنهادات:")
        if not high_corr_pairs and high_vif_df.empty:
            print("✅ هیچ مشکل چندهمخطی‌بودگی شدیدی شناسایی نشد.")
        else:
            if high_corr_pairs:
                print(f"\n🚨 {len(high_corr_pairs)} زوج ستون با همبستگی بالا:")
                for c1, c2, val in high_corr_pairs:
                    print(f"   • {c1} و {c2} (همبستگی = {val:.2f})")
            if not high_vif_df.empty:
                print(f"\n⚠️ {len(high_vif_df)} ستون با VIF بالا:")
                for _, row in high_vif_df.iterrows():
                    print(f"   • {row['feature']} (VIF = {row['VIF']:.2f})")
        
        return df