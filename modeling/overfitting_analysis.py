"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۴ — بررسی تخصصی Overfitting                          ║
║  Phase 14 — Overfitting Analysis                            ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    سنجش دقیق میزان شکاف عملکردی مدل‌ها بین محیط آموزش و تست
    جهت رتبه‌بندی پایداری و قدرت تعمیم‌دهی الگوریتم‌ها.

مدل‌های تحلیل‌شونده:
    ۱. Simple LR
    ۲. Multivariate LR
    ۳. Polynomial
    ۴. Ridge
    ۵. Lasso
    ۶. ElasticNet

خروجی‌ها:
    • جدول تخصصی تحلیل Overfitting (مرتب‌شده بر اساس R² Gap)
    • علامت‌گذاری مدل‌های اورفیت شده
    • گزارش تحلیلی درباره موفقیت Regularization
    • نمودار Grouped Bar Chart (Train vs Test vs CV R²)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.config import C
from modeling.cross_validator import Phase5Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۴
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase14Result:
    """نتایج فاز ۱۴ — تحلیل Overfitting"""
    
    # جدول تحلیل Overfitting
    overfitting_df: pd.DataFrame
    
    # لیست مدل‌های اورفیت شده
    overfitted_models: List[str]
    
    # لیست مدل‌های پایدار
    stable_models: List[str]
    
    # گزارش تحلیلی
    analysis_report: str
    
    # آمار کلی
    n_models_analyzed: int
    n_overfitted: int
    n_stable: int


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی
# ══════════════════════════════════════════════════════════════

def _extract_model_metrics(phase5_result: Phase5Result) -> pd.DataFrame:
    """
    استخراج معیارهای کلیدی از تمام مدل‌ها
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل models_results)
    
    بازگشت:
        DataFrame شامل معیارهای هر مدل
    """
    rows = []
    
    for model_name, metrics in phase5_result.models_results.items():
        row = {
            'Model Name': model_name,
            'Train R²': metrics.get('train_r2', np.nan),
            'Test R²': metrics.get('test_r2', np.nan),
            'CV R² Mean': metrics.get('cv_r2_mean', np.nan),
            'Train MSE': metrics.get('train_mse', np.nan),
            'Test MSE': metrics.get('test_mse', np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # محاسبه R² Gap
    df['R² Gap'] = df['Train R²'] - df['Test R²']
    
    return df


def _identify_overfitted_models(df: pd.DataFrame, threshold: float = 0.1) -> List[str]:
    """
    شناسایی مدل‌های اورفیت شده (R² Gap > threshold)
    
    پارامترها:
        df: DataFrame معیارها
        threshold: آستانه R² Gap (پیش‌فرض: 0.1)
    
    بازگشت:
        لیست نام مدل‌های اورفیت شده
    """
    overfitted = df[df['R² Gap'] > threshold]['Model Name'].tolist()
    return overfitted


def _identify_stable_models(df: pd.DataFrame, threshold: float = 0.1) -> List[str]:
    """
    شناسایی مدل‌های پایدار (R² Gap <= threshold)
    
    پارامترها:
        df: DataFrame معیارها
        threshold: آستانه R² Gap
    
    بازگشت:
        لیست نام مدل‌های پایدار
    """
    stable = df[df['R² Gap'] <= threshold]['Model Name'].tolist()
    return stable


def _generate_analysis_report(df: pd.DataFrame, overfitted_models: List[str]) -> str:
    """
    تولید گزارش تحلیلی درباره موفقیت Regularization
    
    پارامترها:
        df: DataFrame معیارها
        overfitted_models: لیست مدل‌های اورفیت شده
    
    بازگشت:
        رشته گزارش تحلیلی
    """
    report = []
    
    report.append("\n" + "=" * 65)
    report.append("📊  گزارش تحلیلی Overfitting")
    report.append("=" * 65)
    
    # ۱. تحلیل کلی
    report.append(f"\n▶  تحلیل کلی:")
    report.append(f"   • تعداد مدل‌های تحلیل‌شده: {len(df)}")
    report.append(f"   • تعداد مدل‌های اورفیت شده: {len(overfitted_models)}")
    report.append(f"   • تعداد مدل‌های پایدار: {len(df) - len(overfitted_models)}")
    
    # ۲. تحلیل Regularization
    report.append(f"\n▶  تحلیل موفقیت Regularization:")
    
    # مقایسه Multivariate با Ridge
    if "Multivariate LR" in df['Model Name'].values and "Ridge" in df['Model Name'].values:
        multi_gap = df[df['Model Name'] == "Multivariate LR"]['R² Gap'].values[0]
        ridge_gap = df[df['Model Name'] == "Ridge"]['R² Gap'].values[0]
        
        report.append(f"\n   📌 Ridge vs Multivariate:")
        report.append(f"      • Multivariate R² Gap: {multi_gap:.4f}")
        report.append(f"      • Ridge R² Gap       : {ridge_gap:.4f}")
        
        if ridge_gap < multi_gap:
            improvement = multi_gap - ridge_gap
            report.append(f"      ✅ Ridge شکاف را {improvement:.4f} کاهش داد (موفق)")
        else:
            report.append(f"      ⚠️ Ridge شکاف را کاهش نداد")
    
    # مقایسه Multivariate با Lasso
    if "Multivariate LR" in df['Model Name'].values and "Lasso" in df['Model Name'].values:
        multi_gap = df[df['Model Name'] == "Multivariate LR"]['R² Gap'].values[0]
        lasso_gap = df[df['Model Name'] == "Lasso"]['R² Gap'].values[0]
        
        report.append(f"\n   📌 Lasso vs Multivariate:")
        report.append(f"      • Multivariate R² Gap: {multi_gap:.4f}")
        report.append(f"      • Lasso R² Gap       : {lasso_gap:.4f}")
        
        if lasso_gap < multi_gap:
            improvement = multi_gap - lasso_gap
            report.append(f"      ✅ Lasso شکاف را {improvement:.4f} کاهش داد (موفق)")
        else:
            report.append(f"      ⚠️ Lasso شکاف را کاهش نداد")
    
    # مقایسه Multivariate با ElasticNet
    if "Multivariate LR" in df['Model Name'].values and "ElasticNet" in df['Model Name'].values:
        multi_gap = df[df['Model Name'] == "Multivariate LR"]['R² Gap'].values[0]
        en_gap = df[df['Model Name'] == "ElasticNet"]['R² Gap'].values[0]
        
        report.append(f"\n   📌 ElasticNet vs Multivariate:")
        report.append(f"      • Multivariate R² Gap: {multi_gap:.4f}")
        report.append(f"      • ElasticNet R² Gap  : {en_gap:.4f}")
        
        if en_gap < multi_gap:
            improvement = multi_gap - en_gap
            report.append(f"      ✅ ElasticNet شکاف را {improvement:.4f} کاهش داد (موفق)")
        else:
            report.append(f"      ⚠️ ElasticNet شکاف را کاهش نداد")
    
    # ۳. تحلیل Polynomial
    if "Polynomial" in df['Model Name'].values:
        poly_row = df[df['Model Name'].str.startswith("Polynomial")]
        if len(poly_row) > 0:
            poly_gap = poly_row['R² Gap'].values[0]
            report.append(f"\n   📌 Polynomial:")
            report.append(f"      • R² Gap: {poly_gap:.4f}")
            
            if poly_gap > 0.1:
                report.append(f"      ⚠️ Polynomial دچار Overfitting است")
                report.append(f"         (افزایش پیچیدگی منجر به Overfitting شده)")
            else:
                report.append(f"      ✅ Polynomial پایدار است")
    
    # ۴. تحلیل اعتبار CV
    report.append(f"\n▶  تحلیل اعتبار Cross-Validation:")
    
    for _, row in df.iterrows():
        model_name = row['Model Name']
        test_r2 = row['Test R²']
        cv_r2 = row['CV R² Mean']
        
        if not np.isnan(test_r2) and not np.isnan(cv_r2):
            diff = abs(test_r2 - cv_r2)
            if diff < 0.05:
                report.append(f"   ✅ {model_name}: CV R² ({cv_r2:.4f}) ≈ Test R² ({test_r2:.4f}) — CV معتبر")
            else:
                report.append(f"   ⚠️ {model_name}: CV R² ({cv_r2:.4f}) ≠ Test R² ({test_r2:.4f}) — تفاوت {diff:.4f}")
    
    # ۵. نتیجه‌گیری نهایی
    report.append(f"\n▶  نتیجه‌گیری نهایی:")
    
    if len(overfitted_models) == 0:
        report.append(f"   ✅ هیچ مدلی دچار Overfitting نیست")
        report.append(f"   ✅ تمام مدل‌ها تعمیم‌پذیری خوبی دارند")
    elif len(overfitted_models) == 1:
        report.append(f"   ⚠️ یک مدل دچار Overfitting است: {overfitted_models[0]}")
    else:
        report.append(f"   ⚠️ {len(overfitted_models)} مدل دچار Overfitting هستند:")
        for model in overfitted_models:
            report.append(f"      • {model}")
    
    report.append("\n" + "=" * 65)
    
    return "\n".join(report)


def _plot_overfitting_bar_chart(df: pd.DataFrame, save_dir: Path) -> None:
    """
    رسم نمودار Grouped Bar Chart برای تحلیل Overfitting
    
    پارامترها:
        df: DataFrame معیارها (مرتب‌شده بر اساس R² Gap)
        save_dir: مسیر ذخیره نمودار
    """
    # استخراج داده‌ها
    model_names = df['Model Name'].tolist()
    n_models = len(model_names)
    
    # رنگ‌های اختصاصی
    colors = {
        'Train R²': '#E74C3C',      # قرمز
        'Test R²': '#3498DB',       # آبی
        'CV R² Mean': '#2ECC71'     # سبز
    }
    
    # ایجاد figure
    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.2), 7))
    
    # موقعیت میله‌ها
    x = np.arange(n_models)
    width = 0.25  # عرض هر میله
    
    # رسم میله‌ها
    ax.bar(x - width, df['Train R²'], width, 
           label='Train R²', color=colors['Train R²'], alpha=0.85)
    ax.bar(x, df['Test R²'], width, 
           label='Test R²', color=colors['Test R²'], alpha=0.85)
    ax.bar(x + width, df['CV R² Mean'], width, 
           label='CV R² Mean', color=colors['CV R² Mean'], alpha=0.85)
    
    # تنظیمات محور
    ax.set_xlabel('Models (sorted by R² Gap — descending)', fontsize=12, fontweight='normal')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='normal')
    ax.set_title(
        'Phase 14 | Overfitting Analysis — Train vs Test vs CV R²\n'
        '(Wide gap between Train and Test = Overfitting)',
        fontsize=13, fontweight='normal', pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    
    # خط افقی روی R² = 1
    ax.axhline(1, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    
    # راهنما
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # اضافه کردن مقادیر R² Gap بالای هر گروه
    for i, gap in enumerate(df['R² Gap']):
        ax.text(
            i, max(df.iloc[i][['Train R²', 'Test R²', 'CV R² Mean']]) + 0.02,
            f'Gap={gap:.3f}',
            ha='center', fontsize=8, fontweight='bold',
            color='red' if gap > 0.1 else 'green'
        )
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig14_overfitting_analysis.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۱۴
# ══════════════════════════════════════════════════════════════

def run_overfitting_analysis(
    phase5_result: Phase5Result,
    output_folder: str = 'phase14_overfitting_analysis_plots'
) -> Phase14Result:
    """
    اجرای کامل فاز ۱۴ — بررسی تخصصی Overfitting
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل models_results)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase14Result: شامل جدول تحلیل و گزارش
    """
    print("\n" + "=" * 65)
    print("📌  فاز ۱۴ — بررسی تخصصی Overfitting")
    print("=" * 65)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره نمودارها: {save_dir}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: استخراج معیارها
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش اول: استخراج معیارهای کلیدی")
    print(f"{'─' * 65}")
    
    overfitting_df = _extract_model_metrics(phase5_result)
    
    print(f"\n▶  تعداد مدل‌های یافت‌شده: {len(overfitting_df)}")
    for model_name in overfitting_df['Model Name']:
        print(f"   • {model_name}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: محاسبه R² Gap و مرتب‌سازی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📋  بخش دوم: محاسبه R² Gap و مرتب‌سازی")
    print(f"{'─' * 65}")
    
    # مرتب‌سازی نزولی بر اساس R² Gap
    overfitting_df = overfitting_df.sort_values('R² Gap', ascending=False)
    overfitting_df = overfitting_df.reset_index(drop=True)
    
    # گرد کردن اعداد
    numeric_cols = ['Train R²', 'Test R²', 'CV R² Mean', 'R² Gap', 'Train MSE', 'Test MSE']
    for col in numeric_cols:
        overfitting_df[col] = overfitting_df[col].round(4)
    
    print(f"\n▶  جدول تحلیل Overfitting (مرتب‌شده بر اساس R² Gap):")
    print("-" * 120)
    print(overfitting_df.to_string(index=False))
    print("-" * 120)
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: شناسایی مدل‌های اورفیت شده
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🔍  بخش سوم: شناسایی مدل‌های اورفیت شده")
    print(f"{'─' * 65}")
    
    overfitted_models = _identify_overfitted_models(overfitting_df, threshold=0.1)
    stable_models = _identify_stable_models(overfitting_df, threshold=0.1)
    
    print(f"\n▶  مدل‌های اورفیت شده (R² Gap > 0.1):")
    if overfitted_models:
        for model in overfitted_models:
            gap = overfitting_df[overfitting_df['Model Name'] == model]['R² Gap'].values[0]
            print(f"   ⚠️ {model} (R² Gap = {gap:.4f})")
    else:
        print(f"   ✅ هیچ مدلی اورفیت نشده است")
    
    print(f"\n▶  مدل‌های پایدار (R² Gap ≤ 0.1):")
    if stable_models:
        for model in stable_models:
            gap = overfitting_df[overfitting_df['Model Name'] == model]['R² Gap'].values[0]
            print(f"   ✅ {model} (R² Gap = {gap:.4f})")
    else:
        print(f"   ⚠️ هیچ مدل پایداری وجود ندارد")
    
    # ══════════════════════════════════════════════════════════
    #  بخش چهارم: گزارش تحلیلی
    # ══════════════════════════════════════════════════════════
    analysis_report = _generate_analysis_report(overfitting_df, overfitted_models)
    print(analysis_report)
    
    # ══════════════════════════════════════════════════════════
    #  ساخت نتیجه
    # ══════════════════════════════════════════════════════════
    result = Phase14Result(
        overfitting_df=overfitting_df,
        overfitted_models=overfitted_models,
        stable_models=stable_models,
        analysis_report=analysis_report,
        n_models_analyzed=len(overfitting_df),
        n_overfitted=len(overfitted_models),
        n_stable=len(stable_models)
    )
    
    # ══════════════════════════════════════════════════════════
    #  رسم نمودار
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📈  رسم نمودار Grouped Bar Chart...")
    print(f"{'─' * 65}")
    
    _plot_overfitting_bar_chart(overfitting_df, save_dir)
    
    # ── پایان فاز ۱۴ ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  فاز ۱۴ — بررسی تخصصی Overfitting کامل شد.")
    print(f"   • مدل‌های تحلیل‌شده: {result.n_models_analyzed}")
    print(f"   • مدل‌های اورفیت شده: {result.n_overfitted}")
    print(f"   • مدل‌های پایدار: {result.n_stable}")
    print("=" * 65)
    
    return result