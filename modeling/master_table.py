"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۵ — جدول مقایسه نهایی جامع (Master Comparison Table) ║
║  Phase 15 — Master Comparison Table                         ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    تجمیع تمام نتایج، آمارها، معیارها و پارامترهای محاسباتی کل
    مدل‌های پروژه در یک بانک اطلاعاتی یکپارچه جهت اتخاذ تصمیم نهایی.

خروجی‌ها:
    • جدول جامع DataFrame با ۱۲ ستون
    • مرتب‌سازی بر اساس Test R² (نزولی)
    • ذخیره در فایل CSV (regression_models_report.csv)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cross_validator import Phase5Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۵
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase15Result:
    """نتایج فاز ۱۵ — جدول مقایسه نهایی"""
    
    # جدول جامع
    master_df: pd.DataFrame
    
    # مسیر فایل CSV ذخیره‌شده
    csv_path: Path
    
    # بهترین مدل (بر اساس Test R²)
    best_model_name: str
    best_model_r2: float
    
    # آمار کلی
    n_models: int


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی
# ══════════════════════════════════════════════════════════════

def _extract_best_params(model_name: str, model_results: dict) -> str:
    """
    استخراج پارامترهای بهینه هر مدل
    
    پارامترها:
        model_name: نام مدل
        model_results: دیکشنری نتایج مدل
    
    بازگشت:
        رشته پارامترهای بهینه
    """
    # Simple LR
    if model_name == "Simple LR":
        return "None"
    
    # Multivariate LR
    elif model_name == "Multivariate LR":
        return "None"
    
    # Polynomial
    elif model_name.startswith("Polynomial"):
        degree = model_results.get('degree', 'N/A')
        return f"degree={degree}"
    
    # Ridge
    elif model_name == "Ridge":
        alpha = model_results.get('best_alpha', 'N/A')
        if isinstance(alpha, (int, float)):
            return f"alpha={alpha:.4f}"
        return "N/A"
    
    # Lasso
    elif model_name == "Lasso":
        alpha = model_results.get('best_alpha', 'N/A')
        if isinstance(alpha, (int, float)):
            return f"alpha={alpha:.4f}"
        return "N/A"
    
    # ElasticNet
    elif model_name == "ElasticNet":
        alpha = model_results.get('best_alpha', 'N/A')
        l1_ratio = model_results.get('best_l1_ratio', 'N/A')
        if isinstance(alpha, (int, float)) and isinstance(l1_ratio, (int, float)):
            return f"alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}"
        return "N/A"
    
    else:
        return "N/A"


def _build_master_table(phase5_result: Phase5Result) -> pd.DataFrame:
    """
    ساخت جدول مقایسه نهایی جامع
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل models_results)
    
    بازگشت:
        DataFrame جدول جامع
    """
    rows = []
    
    for model_name, metrics in phase5_result.models_results.items():
        # استخراج Best Params
        best_params = _extract_best_params(model_name, metrics)
        
        row = {
            'Model Name': model_name,
            'Test MSE': metrics.get('test_mse', np.nan),
            'Test RMSE': metrics.get('test_rmse', np.nan),
            'Test MAE': metrics.get('test_mae', np.nan),
            'Test R²': metrics.get('test_r2', np.nan),
            'Train R²': metrics.get('train_r2', np.nan),
            'CV MSE Mean': metrics.get('cv_mse_mean', np.nan),
            'CV MSE Std': metrics.get('cv_mse_std', np.nan),
            'CV R² Mean': metrics.get('cv_r2_mean', np.nan),
            'CV R² Std': metrics.get('cv_r2_std', np.nan),
            'Best Params': best_params,
            'Training Time (Seconds)': metrics.get('train_time', np.nan),
            'Prediction Time (Seconds)': metrics.get('pred_time', np.nan),
        }
        rows.append(row)
    
    # ساخت DataFrame
    df = pd.DataFrame(rows)
    
    # مرتب‌سازی بر اساس Test R² (نزولی)
    df = df.sort_values('Test R²', ascending=False)
    df = df.reset_index(drop=True)
    
    # گرد کردن اعداد اعشاری تا ۴ رقم
    numeric_cols = [
        'Test MSE', 'Test RMSE', 'Test MAE', 'Test R²', 'Train R²',
        'CV MSE Mean', 'CV MSE Std', 'CV R² Mean', 'CV R² Std',
        'Training Time (Seconds)', 'Prediction Time (Seconds)'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    return df


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۱۵
# ══════════════════════════════════════════════════════════════

def run_master_table(
    phase5_result: Phase5Result,
    output_folder: str = 'phase15_master_table'
) -> Phase15Result:
    """
    اجرای کامل فاز ۱۵ — جدول مقایسه نهایی
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل models_results)
        output_folder: نام پوشه ذخیره فایل CSV
    
    بازگشت:
        Phase15Result: شامل جدول جامع و مسیر فایل
    """
    print("\n" + "=" * 65)
    print("📌  فاز ۱۵ — جدول مقایسه نهایی جامع (Master Comparison Table)")
    print("=" * 65)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره فایل: {save_dir}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: ساخت جدول جامع
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش اول: جمع‌آوری و سازماندهی شاخص‌ها")
    print(f"{'─' * 65}")
    
    master_df = _build_master_table(phase5_result)
    
    n_models = len(master_df)
    print(f"\n▶  تعداد مدل‌های جمع‌آوری‌شده: {n_models}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: نمایش جدول نهایی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📋  بخش دوم: جدول مقایسه نهایی (مرتب‌شده بر اساس Test R²)")
    print(f"{'─' * 65}")
    
    # تنظیمات نمایش pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)
    
    print(f"\n{master_df.to_string(index=False)}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: شناسایی بهترین مدل
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🏆  بخش سوم: شناسایی بهترین مدل")
    print(f"{'─' * 65}")
    
    best_model_name = master_df.iloc[0]['Model Name']
    best_model_r2 = master_df.iloc[0]['Test R²']
    
    print(f"\n   🥇 بهترین مدل (بر اساس Test R²):")
    print(f"      • نام مدل    : {best_model_name}")
    print(f"      • Test R²    : {best_model_r2:.4f}")
    print(f"      • Best Params: {master_df.iloc[0]['Best Params']}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش چهارم: ذخیره فایل CSV
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"💾  بخش چهارم: ذخیره فایل CSV")
    print(f"{'─' * 65}")
    
    csv_path = save_dir / 'regression_models_report.csv'
    master_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n   ✅ فایل CSV با موفقیت ذخیره شد:")
    print(f"      • مسیر: {csv_path}")
    print(f"      • نام فایل: {csv_path.name}")
    print(f"      • تعداد ردیف‌ها: {n_models}")
    print(f"      • تعداد ستون‌ها: {len(master_df.columns)}")
    
    # ══════════════════════════════════════════════════════════
    #  ساخت نتیجه
    # ══════════════════════════════════════════════════════════
    result = Phase15Result(
        master_df=master_df,
        csv_path=csv_path,
        best_model_name=best_model_name,
        best_model_r2=best_model_r2,
        n_models=n_models
    )
    
    # ── پایان فاز ۱۵ ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  فاز ۱۵ — جدول مقایسه نهایی کامل شد.")
    print(f"   • تعداد مدل‌ها: {n_models}")
    print(f"   • بهترین مدل: {best_model_name} (R² = {best_model_r2:.4f})")
    print(f"   • فایل CSV: {csv_path.name}")
    print("=" * 65)
    
    return result