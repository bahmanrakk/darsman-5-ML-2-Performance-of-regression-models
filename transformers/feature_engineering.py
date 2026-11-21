import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    ترانسفورمر جامع مهندسی ویژگی
    اجرای ۱۱ تابع برای ایجاد ویژگی‌های جدید از داده‌های خودرو
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        print("\n" + "=" * 60)
        print("🔧 مرحله مهندسی ویژگی (Feature Engineering)")
        print("=" * 60)
        
        # اجرای ۱۱ تابع به ترتیب
        df = self._fe_annual_km(df)
        df = self._fe_torque_power(df)
        df = self._fe_power_ratio(df)
        df = self._fe_power_per_cc(df)
        df = self._fe_torque_per_cc(df)
        df = self._fe_mileage_per_cc(df)
        df = self._fe_age_squared(df)
        df = self._fe_wear_index(df)
        df = self._fe_km_per_seat(df)
        df = self._fe_premium_seller(df)
        df = self._fe_auto_high_power(df)
        
        print("\n✅ تمام ویژگی‌های مهندسی‌شده با موفقیت ایجاد شدند")
        print(f"📊 تعداد ستون‌های نهایی: {len(df.columns)}")
        
        return df
    
    # ------------------------------------------------------------------
    # 1. محاسبه کیلومتر سالانه
    # ------------------------------------------------------------------
    def _fe_annual_km(self, df):
        print("\n📊 1. محاسبه کیلومتر سالانه...")
        
        required_columns = ['km_driven', 'age', 'age_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        # اعتبارسنجی age
        if (df['age'] < 0).any():
            raise ValueError(f"ستون 'age' حاوی مقادیر منفی است")
        if (df['age'] == 0).any():
            df['age'] = df['age'] + 1
        
        # اعتبارسنجی age_category
        if (df['age_category'] < 0).any():
            raise ValueError(f"ستون 'age_category' حاوی مقادیر منفی است")
        if (df['age_category'] == 0).any():
            df['age_category'] = df['age_category'] + 1
        
        # محاسبه
        df['annual_km'] = df['km_driven'] / df['age']
        df['annual_km_per_age_category'] = df['km_driven'] / df['age_category']
        
        print(f"   ✅ annual_km و annual_km_per_age_category ایجاد شدند")
        return df
    
    # ------------------------------------------------------------------
    # 2. محاسبه torque_power
    # ------------------------------------------------------------------
    def _fe_torque_power(self, df):
        print("\n📊 2. محاسبه torque_power...")
        
        required_columns = ['torque_nm', 'torque_rpm']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['torque_nm'] < 0).any():
            raise ValueError(f"ستون 'torque_nm' حاوی مقادیر منفی است")
        if (df['torque_rpm'] < 0).any():
            raise ValueError(f"ستون 'torque_rpm' حاوی مقادیر منفی است")
        
        # محاسبه با فرمول فیزیکی
        df['torque_power'] = (
            (df['torque_nm'] * 2 * math.pi * df['torque_rpm']) / 60000
        ).round(3)
        
        print(f"   ✅ torque_power محاسبه شد")
        print(f"   میانگین: {df['torque_power'].mean():.3f} kW")
        return df
    
    # ------------------------------------------------------------------
    # 3. محاسبه power_ratio
    # ------------------------------------------------------------------
    def _fe_power_ratio(self, df):
        print("\n📊 3. محاسبه power_ratio...")
        
        required_columns = ['torque_power', 'max_power']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['max_power'] < 0).any():
            raise ValueError(f"ستون 'max_power' حاوی مقادیر منفی است")
        
        eps = 1e-9
        df['power_ratio'] = df['torque_power'] / (df['max_power'] + eps)
        
        inconsistent_count = ((df['power_ratio'] < 0.95) | (df['power_ratio'] > 1.05)).sum()
        if inconsistent_count > 0:
            print(f"   ⚠️ {inconsistent_count} رکورد ناسازگار شناسایی شد")
        
        print(f"   ✅ power_ratio محاسبه شد")
        return df
    
    # ------------------------------------------------------------------
    # 4. محاسبه power_per_cc
    # ------------------------------------------------------------------
    def _fe_power_per_cc(self, df):
        print("\n📊 4. محاسبه power_per_cc...")
        
        required_columns = ['max_power', 'engine']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['engine'] < 0).any():
            raise ValueError(f"ستون 'engine' حاوی مقادیر منفی است")
        
        # جایگزینی مقادیر بسیار کوچک
        min_engine = 100
        df['engine'] = np.where(df['engine'] < min_engine, min_engine, df['engine'])
        
        if (df['max_power'] < 0).any():
            raise ValueError(f"ستون 'max_power' حاوی مقادیر منفی است")
        
        df['power_per_cc'] = df['max_power'] / df['engine']
        
        if 'engine_category_encoded' in df.columns:
            if (df['engine_category_encoded'] < 0).any():
                raise ValueError(f"ستون 'engine_category_encoded' حاوی مقادیر منفی است")
            df['engine_category_encoded'] = np.where(
                df['engine_category_encoded'] < 1, 1, df['engine_category_encoded']
            )
            df['power_per_category'] = df['max_power'] / df['engine_category_encoded']
        
        print(f"   ✅ power_per_cc محاسبه شد")
        return df
    
    # ------------------------------------------------------------------
    # 5. محاسبه torque_per_cc
    # ------------------------------------------------------------------
    def _fe_torque_per_cc(self, df):
        print("\n📊 5. محاسبه torque_per_cc...")
        
        required_columns = ['torque_nm', 'engine']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['engine'] < 0).any():
            raise ValueError(f"ستون 'engine' حاوی مقادیر منفی است")
        
        min_engine = 100
        df['engine'] = np.where(df['engine'] < min_engine, min_engine, df['engine'])
        
        df['torque_per_cc'] = df['torque_nm'] / df['engine']
        
        if 'engine_category_encoded' in df.columns:
            if (df['engine_category_encoded'] < 0).any():
                raise ValueError(f"ستون 'engine_category_encoded' حاوی مقادیر منفی است")
            df['engine_category_encoded'] = np.where(
                df['engine_category_encoded'] < 1, 1, df['engine_category_encoded']
            )
            df['torque_per_category'] = df['torque_nm'] / df['engine_category_encoded']
        
        print(f"   ✅ torque_per_cc محاسبه شد")
        return df
    
    # ------------------------------------------------------------------
    # 6. محاسبه mileage_per_cc
    # ------------------------------------------------------------------
    def _fe_mileage_per_cc(self, df):
        print("\n📊 6. محاسبه mileage_per_cc...")
        
        required_columns = ['mileage', 'engine']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['engine'] < 0).any():
            raise ValueError(f"ستون 'engine' حاوی مقادیر منفی است")
        
        min_engine = 100
        df['engine'] = np.where(df['engine'] < min_engine, min_engine, df['engine'])
        
        if (df['mileage'] < 0).any():
            raise ValueError(f"ستون 'mileage' حاوی مقادیر منفی است")
        
        df['mileage_per_cc'] = df['mileage'] / df['engine']
        
        if 'engine_category_encoded' in df.columns:
            if (df['engine_category_encoded'] < 0).any():
                raise ValueError(f"ستون 'engine_category_encoded' حاوی مقادیر منفی است")
            df['engine_category_encoded'] = np.where(
                df['engine_category_encoded'] < 1, 1, df['engine_category_encoded']
            )
            df['mileage_per_category'] = df['mileage'] / df['engine_category_encoded']
        
        print(f"   ✅ mileage_per_cc محاسبه شد")
        return df
    
    # ------------------------------------------------------------------
    # 7. محاسبه age_squared
    # ------------------------------------------------------------------
    def _fe_age_squared(self, df):
        print("\n📊 7. محاسبه age_squared...")
        
        required_columns = ['age', 'age_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['age'] < 0).any():
            raise ValueError(f"ستون 'age' حاوی مقادیر منفی است")
        if (df['age_category'] < 0).any():
            raise ValueError(f"ستون 'age_category' حاوی مقادیر منفی است")
        
        df['age_squared'] = df['age'] ** 2
        df['age_category_squared'] = df['age_category'] ** 2
        
        print(f"   ✅ age_squared و age_category_squared محاسبه شدند")
        return df
    
    # ------------------------------------------------------------------
    # 8. محاسبه wear_index
    # ------------------------------------------------------------------
    def _fe_wear_index(self, df):
        print("\n📊 8. محاسبه wear_index...")
        
        required_columns = ['age', 'km_driven', 'age_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        for col in ['age', 'age_category', 'km_driven']:
            if (df[col] < 0).any():
                raise ValueError(f"ستون '{col}' حاوی مقادیر منفی است")
        
        df['wear_index'] = df['age'] * df['km_driven']
        df['wear_index_category'] = df['age_category'] * df['km_driven']
        
        print(f"   ✅ wear_index و wear_index_category محاسبه شدند")
        return df
    
    # ------------------------------------------------------------------
    # 9. محاسبه km_per_seat
    # ------------------------------------------------------------------
    def _fe_km_per_seat(self, df):
        print("\n📊 9. محاسبه km_per_seat...")
        
        required_columns = ['km_driven', 'seats', 'seats_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['km_driven'] < 0).any():
            raise ValueError(f"ستون 'km_driven' حاوی مقادیر منفی است")
        if (df['seats'] < 0).any():
            raise ValueError(f"ستون 'seats' حاوی مقادیر منفی است")
        if (df['seats_category'] < 0).any():
            raise ValueError(f"ستون 'seats_category' حاوی مقادیر منفی است")
        
        # جایگزینی صفر با 1
        df['seats'] = np.where(df['seats'] == 0, 1, df['seats'])
        df['seats_category'] = np.where(df['seats_category'] == 0, 1, df['seats_category'])
        
        eps = 1e-9
        df['km_per_seat'] = df['km_driven'] / (df['seats'] + eps)
        df['km_per_seat_category'] = df['km_driven'] / (df['seats_category'] + eps)
        
        print(f"   ✅ km_per_seat و km_per_seat_category محاسبه شدند")
        return df
    
    # ------------------------------------------------------------------
    # 10. محاسبه premium_seller
    # ------------------------------------------------------------------
    def _fe_premium_seller(self, df):
        print("\n📊 10. محاسبه premium_seller...")
        
        required_columns = ['seller_type_Dealer', 'seller_type_Individual', 
                           'seller_type_Trustmark_Dealer', 'age', 'age_category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        if (df['age'] < 0).any():
            raise ValueError(f"ستون 'age' حاوی مقادیر منفی است")
        if (df['age_category'] < 0).any():
            raise ValueError(f"ستون 'age_category' حاوی مقادیر منفی است")
        
        # محاسبه seller_type_score
        df['seller_type_score'] = (
            df['seller_type_Individual'] * 1 +
            df['seller_type_Dealer'] * 2 +
            df['seller_type_Trustmark_Dealer'] * 3
        )
        
        # جایگزینی مقادیر نامعتبر
        invalid_scores = (~df['seller_type_score'].isin([1, 2, 3])).sum()
        if invalid_scores > 0:
            median_score = df['seller_type_score'].median()
            df['seller_type_score'] = np.where(
                df['seller_type_score'].isin([1, 2, 3]),
                df['seller_type_score'],
                median_score
            )
        
        df['premium_seller'] = df['seller_type_score'] / (df['age'] + 1)
        df['premium_seller_category'] = df['seller_type_score'] / (df['age_category'] + 1)
        
        # حذف ستون موقت
        df = df.drop(columns=['seller_type_score'])
        
        print(f"   ✅ premium_seller و premium_seller_category محاسبه شدند")
        return df
    
    # ------------------------------------------------------------------
    # 11. محاسبه auto_high_power
    # ------------------------------------------------------------------
    def _fe_auto_high_power(self, df):
        print("\n📊 11. محاسبه auto_high_power...")
        
        required_columns = ['transmission_bin', 'max_power']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون '{col}' وجود ندارد")
        
        MANUAL_COEFFICIENT = 0.85
        AUTOMATIC_COEFFICIENT = 1.15
        
        # اعتبارسنجی transmission_bin
        unique_values = df['transmission_bin'].unique()
        valid_values = [0, 1]
        invalid_values = [v for v in unique_values if v not in valid_values]
        if invalid_values:
            raise ValueError(f"ستون 'transmission_bin' حاوی مقادیر نامعتبر {invalid_values} است")
        
        if (df['max_power'] < 0).any():
            raise ValueError(f"ستون 'max_power' حاوی مقادیر منفی است")
        
        # ایجاد ضریب گیربکس
        df['transmission_coefficient'] = np.where(
            df['transmission_bin'] == 0,
            MANUAL_COEFFICIENT,
            AUTOMATIC_COEFFICIENT
        )
        
        df['auto_high_power'] = df['transmission_coefficient'] * df['max_power']
        
        # حذف ستون موقت
        df = df.drop(columns=['transmission_coefficient'])
        
        print(f"   ✅ auto_high_power محاسبه شد")
        return df