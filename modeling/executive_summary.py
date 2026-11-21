"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۷ — تحلیل خودکار و نتیجه‌گیری نهایی                  ║
║  Phase 17 — Executive Summary & Final Recommendation        ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    فراتر رفتن از نمایش صرف اعداد و نمودارها؛ تولید یک سیستم
    گزارش‌دهی هوشمند و تحلیل کیفی-مفهومی برای ارزیابی نهایی
    و ارائه توصیه کاربردی.

سه بخش اصلی:
    ۱. Executive Summary — خلاصه خودکار بهترین‌ها
    ۲. Conceptual Matrix — جدول مقایسه مفهومی مدل‌ها
    ۳. Final Recommendation — تحلیل متنی و توصیه نهایی
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cross_validator import Phase5Result
from modeling.master_table import Phase15Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۷
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase17Result:
    """نتایج فاز ۱۷ — تحلیل نهایی"""
    
    # بخش اول: خلاصه بهترین‌ها
    best_models: Dict[str, str] = field(default_factory=dict)
    
    # بخش دوم: جدول مفهومی
    conceptual_matrix: pd.DataFrame = None
    
    # بخش سوم: توصیه نهایی
    final_recommendation: str = ""
    
    # مدل برتر نهایی
    overall_best_model: str = ""
    overall_best_r2: float = 0.0


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | بخش اول — خلاصه خودکار بهترین‌ها
# ══════════════════════════════════════════════════════════════

def _extract_best_models(master_df: pd.DataFrame,
                        phase5_result: Phase5Result) -> Dict[str, str]:
    """
    استخراج خودکار بهترین مدل‌ها بر اساس معیارهای مختلف
    
    پارامترها:
        master_df: جدول مقایسه نهایی از فاز ۱۵
        phase5_result: نتیجه فاز ۵ (شامل اطلاعات اضافی)
    
    بازگشت:
        دیکشنری شامل بهترین مدل برای هر معیار
    """
    best = {}
    
    # ── معیارهای عددی (از master_df) ─────────────────────────
    
    # بهترین Test MSE (کمترین)
    best['Test MSE'] = master_df.loc[master_df['Test MSE'].idxmin(), 'Model Name']
    
    # بهترین Test RMSE (کمترین)
    best['Test RMSE'] = master_df.loc[master_df['Test RMSE'].idxmin(), 'Model Name']
    
    # بهترین Test MAE (کمترین)
    best['Test MAE'] = master_df.loc[master_df['Test MAE'].idxmin(), 'Model Name']
    
    # بهترین Test R² (بیشترین)
    best['Test R²'] = master_df.loc[master_df['Test R²'].idxmax(), 'Model Name']
    
    # بهترین CV MSE Mean (کمترین)
    best['CV MSE Mean'] = master_df.loc[master_df['CV MSE Mean'].idxmin(), 'Model Name']
    
    # بهترین CV R² Mean (بیشترین)
    best['CV R² Mean'] = master_df.loc[master_df['CV R² Mean'].idxmax(), 'Model Name']
    
    # پایدارترین مدل (کمترین CV MSE Std)
    best['Stability (CV MSE Std)'] = master_df.loc[master_df['CV MSE Std'].idxmin(), 'Model Name']
    
    # سریع‌ترین در آموزش (کمترین Training Time)
    best['Training Time'] = master_df.loc[master_df['Training Time (Seconds)'].idxmin(), 'Model Name']
    
    # سریع‌ترین در پیش‌بینی (کمترین Prediction Time)
    best['Prediction Time'] = master_df.loc[master_df['Prediction Time (Seconds)'].idxmin(), 'Model Name']
    
    # ── اطلاعات خاص مدل‌ها (از phase5_result) ────────────────
    
    # بهترین درجه Polynomial
    for model_name in master_df['Model Name']:
        if model_name.startswith('Polynomial'):
            degree = phase5_result.models_results[model_name].get('degree', 'N/A')
            best['Polynomial Degree'] = f"{model_name} (degree={degree})"
            break
    
    # بهترین alpha برای Ridge
    if 'Ridge' in phase5_result.models_results:
        alpha = phase5_result.models_results['Ridge'].get('best_alpha', 'N/A')
        best['Ridge Alpha'] = f"α = {alpha:.6f}" if isinstance(alpha, (int, float)) else alpha
    
    # بهترین alpha برای Lasso
    if 'Lasso' in phase5_result.models_results:
        alpha = phase5_result.models_results['Lasso'].get('best_alpha', 'N/A')
        best['Lasso Alpha'] = f"α = {alpha:.6f}" if isinstance(alpha, (int, float)) else alpha
    
    # بهترین ترکیب برای ElasticNet
    if 'ElasticNet' in phase5_result.models_results:
        alpha = phase5_result.models_results['ElasticNet'].get('best_alpha', 'N/A')
        l1_ratio = phase5_result.models_results['ElasticNet'].get('best_l1_ratio', 'N/A')
        if isinstance(alpha, (int, float)) and isinstance(l1_ratio, (int, float)):
            best['ElasticNet Params'] = f"α = {alpha:.6f}, l1_ratio = {l1_ratio:.2f}"
        else:
            best['ElasticNet Params'] = f"α = {alpha}, l1_ratio = {l1_ratio}"
    
    # ویژگی‌های انتخاب‌شده توسط Lasso
    if 'Lasso' in phase5_result.models_results:
        selected = phase5_result.models_results['Lasso'].get('selected_features', [])
        zeroed = phase5_result.models_results['Lasso'].get('zeroed_features', [])
        best['Lasso Selected Features'] = f"{len(selected)} ویژگی: {', '.join(selected) if selected else 'None'}"
        best['Lasso Zeroed Features'] = f"{len(zeroed)} ویژگی: {', '.join(zeroed) if zeroed else 'None'}"
    
    return best


def _print_executive_summary(best_models: Dict[str, str]) -> None:
    """
    چاپ خلاصه خودکار بهترین‌ها با فرمت زیبا
    """
    print("\n" + "=" * 70)
    print("🏆  بخش اول — خلاصه خودکار بهترین‌ها (Executive Summary)")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    🥇 بهترین مدل‌های پروژه                         │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # معیارهای اصلی
    main_metrics = [
        ('Test MSE (کمترین خطا)', 'Test MSE'),
        ('Test RMSE', 'Test RMSE'),
        ('Test MAE', 'Test MAE'),
        ('Test R² (بیشترین دقت)', 'Test R²'),
        ('CV MSE Mean (اعتبارسنجی متقابل)', 'CV MSE Mean'),
        ('CV R² Mean (توان تبیین CV)', 'CV R² Mean'),
        ('پایدارترین مدل (کمترین Std)', 'Stability (CV MSE Std)'),
        ('سریع‌ترین در آموزش', 'Training Time'),
        ('سریع‌ترین در پیش‌بینی', 'Prediction Time'),
    ]
    
    for title, key in main_metrics:
        if key in best_models:
            value = best_models[key]
            print(f"│  🥇 {title:<35} → {value:<25} │")
    
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # پارامترهای بهینه
    param_metrics = [
        ('بهترین درجه چندجمله‌ای', 'Polynomial Degree'),
        ('بهترین α در Ridge', 'Ridge Alpha'),
        ('بهترین α در Lasso', 'Lasso Alpha'),
        ('بهترین ترکیب ElasticNet', 'ElasticNet Params'),
    ]
    
    for title, key in param_metrics:
        if key in best_models:
            value = best_models[key]
            # کوتاه کردن اگر خیلی بلند باشد
            if len(str(value)) > 30:
                value = str(value)[:30] + "..."
            print(f"│  🎯 {title:<35} → {value:<25} │")
    
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # ویژگی‌های Lasso
    if 'Lasso Selected Features' in best_models:
        print(f"│  📌 ویژگی‌های انتخاب‌شده Lasso:                                │")
        print(f"│     {best_models['Lasso Selected Features']:<65} │")
    
    if 'Lasso Zeroed Features' in best_models:
        print(f"│  📌 ویژگی‌های حذف‌شده Lasso:                                    │")
        print(f"│     {best_models['Lasso Zeroed Features']:<65} │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | بخش دوم — جدول مقایسه مفهومی مدل‌ها
# ══════════════════════════════════════════════════════════════

def _build_conceptual_matrix() -> pd.DataFrame:
    """
    ساخت جدول مقایسه مفهومی مدل‌ها (ثابت)
    
    بازگشت:
        DataFrame شامل ویژگی‌های ماهیتی هر مدل
    """
    data = {
        'مدل رگرسیونی': [
            'خطی تک‌متغیره (Simple LR)',
            'خطی چندمتغیره (Multivariate LR)',
            'چندجمله‌ای (Polynomial)',
            'ریج (Ridge)',
            'لاسو (Lasso)',
            'الاستیک نت (Elastic Net)'
        ],
        'توانایی کشف روابط غیرخطی': [
            '❌ خیر',
            '❌ خیر',
            '✅ بله',
            '❌ خیر',
            '❌ خیر',
            '❌ خیر'
        ],
        'مکانیزم Regularization': [
            '❌ ندارد',
            '❌ ندارد',
            '❌ ندارد',
            '✅ بله (L₂)',
            '✅ بله (L₁)',
            '✅ بله (L₁ + L₂)'
        ],
        'قابلیت انتخاب ویژگی': [
            '❌ خیر',
            '❌ خیر',
            '❌ خیر',
            '❌ خیر',
            '✅ بله',
            '✅ بله (نسبی)'
        ],
        'هایپرپارامترهای تنظیمی': [
            '❌ ندارد',
            '❌ ندارد',
            'درجه چندجمله‌ای (Degree)',
            'α (شدت جریمه)',
            'α (شدت جریمه)',
            'α و l1_ratio'
        ],
        'سطح تفسیرپذیری': [
            '⭐⭐⭐⭐⭐ (عالی)',
            '⭐⭐⭐⭐ (بسیار خوب)',
            '⭐⭐⭐ (متوسط)',
            '⭐⭐⭐⭐ (بسیار خوب)',
            '⭐⭐⭐⭐ (بسیار خوب)',
            '⭐⭐⭐⭐ (بسیار خوب)'
        ],
        'بهترین سناریوی کاربردی': [
            'Baseline اولیه و درک شهودی',
            'سیستم‌های خطی ساده بدون هم‌خطی',
            'روابط غیرخطی در ابعاد پایین',
            'کنترل Overfitting با هم‌خطی شدید',
            'غربالگری ویژگی‌های زیاد',
            'ویژگی‌های زیاد + گروه‌های هم‌خط'
        ]
    }
    
    return pd.DataFrame(data)


def _print_conceptual_matrix(conceptual_df: pd.DataFrame) -> None:
    """
    چاپ جدول مقایسه مفهومی با فرمت زیبا
    """
    print("\n" + "=" * 70)
    print("📋  بخش دوم — جدول مقایسه مفهومی مدل‌ها (Conceptual Matrix)")
    print("=" * 70)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 35)
    
    print(f"\n{conceptual_df.to_string(index=False)}")


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | بخش سوم — تحلیل متنی و توصیه نهایی
# ══════════════════════════════════════════════════════════════

def _generate_final_recommendation(master_df: pd.DataFrame,
                                  phase5_result: Phase5Result,
                                  best_models: Dict[str, str]) -> str:
    """
    تولید تحلیل متنی و توصیه نهایی بر اساس داده‌های واقعی
    
    پارامترها:
        master_df: جدول مقایسه نهایی
        phase5_result: نتیجه فاز ۵
        best_models: دیکشنری بهترین مدل‌ها
    
    بازگشت:
        رشته تحلیل متنی
    """
    report = []
    
    report.append("\n" + "=" * 70)
    report.append("🎯  بخش سوم — تحلیل متنی و توصیه نهایی (Final Recommendation)")
    report.append("=" * 70)
    
    # ── محور ۱: تحلیل Regularization ─────────────────────────
    report.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    report.append("│  📊 محور ۱: تحلیل کارکرد Regularization                            │")
    report.append("└─────────────────────────────────────────────────────────────────────┘")
    
    if all(m in master_df['Model Name'].values for m in ['Multivariate LR', 'Ridge', 'Lasso', 'ElasticNet']):
        multi_r2 = master_df[master_df['Model Name'] == 'Multivariate LR']['Test R²'].values[0]
        ridge_r2 = master_df[master_df['Model Name'] == 'Ridge']['Test R²'].values[0]
        lasso_r2 = master_df[master_df['Model Name'] == 'Lasso']['Test R²'].values[0]
        en_r2 = master_df[master_df['Model Name'] == 'ElasticNet']['Test R²'].values[0]
        
        multi_gap = master_df[master_df['Model Name'] == 'Multivariate LR']['R² Gap'].values[0] if 'R² Gap' in master_df.columns else 0
        ridge_gap = master_df[master_df['Model Name'] == 'Ridge']['R² Gap'].values[0] if 'R² Gap' in master_df.columns else 0
        
        report.append(f"\n   • Multivariate LR (بدون Regularization):")
        report.append(f"      Test R² = {multi_r2:.4f}")
        
        report.append(f"\n   • Ridge (L₂ Regularization):")
        report.append(f"      Test R² = {ridge_r2:.4f}")
        if ridge_r2 > multi_r2:
            report.append(f"      ✅ بهبود R²: +{(ridge_r2 - multi_r2):.4f}")
        else:
            report.append(f"      ⚠️ کاهش R²: {(ridge_r2 - multi_r2):.4f}")
        
        report.append(f"\n   • Lasso (L₁ Regularization):")
        report.append(f"      Test R² = {lasso_r2:.4f}")
        if lasso_r2 > multi_r2:
            report.append(f"      ✅ بهبود R²: +{(lasso_r2 - multi_r2):.4f}")
        else:
            report.append(f"      ⚠️ کاهش R²: {(lasso_r2 - multi_r2):.4f}")
        
        report.append(f"\n   • ElasticNet (L₁ + L₂):")
        report.append(f"      Test R² = {en_r2:.4f}")
        if en_r2 > multi_r2:
            report.append(f"      ✅ بهبود R²: +{(en_r2 - multi_r2):.4f}")
        else:
            report.append(f"      ⚠️ کاهش R²: {(en_r2 - multi_r2):.4f}")
        
        # نتیجه‌گیری
        report.append(f"\n   📌 نتیجه‌گیری:")
        best_reg = max([ridge_r2, lasso_r2, en_r2])
        if best_reg > multi_r2:
            report.append(f"      ✅ Regularization موفق به بهبود مدل شد")
            report.append(f"      بهترین مدل Regularized: {best_models.get('Test R²', 'N/A')}")
        else:
            report.append(f"      ⚠️ Regularization بهبود قابل توجهی نداشت")
    
    # ── محور ۲: بهترین موازنه دقت و پایداری ──────────────────
    report.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    report.append("│  ⚖️  محور ۲: بهترین موازنه دقت و پایداری                          │")
    report.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # محاسبه امتیاز ترکیبی (R² بالا + Std پایین)
    master_df_copy = master_df.copy()
    master_df_copy['R2_Score'] = master_df_copy['Test R²']
    master_df_copy['Stability_Score'] = 1 / (1 + master_df_copy['CV MSE Std'])  # معکوس Std
    master_df_copy['Combined_Score'] = master_df_copy['R2_Score'] * 0.7 + master_df_copy['Stability_Score'] * 0.3
    
    best_balanced = master_df_copy.loc[master_df_copy['Combined_Score'].idxmax(), 'Model Name']
    best_balanced_r2 = master_df_copy.loc[master_df_copy['Combined_Score'].idxmax(), 'Test R²']
    best_balanced_std = master_df_copy.loc[master_df_copy['Combined_Score'].idxmax(), 'CV MSE Std']
    
    report.append(f"\n   🏆 بهترین موازنه دقت و پایداری:")
    report.append(f"      • مدل: {best_balanced}")
    report.append(f"      • Test R²: {best_balanced_r2:.4f}")
    report.append(f"      • CV MSE Std: {best_balanced_std:.2f}")
    report.append(f"      • امتیاز ترکیبی: {master_df_copy['Combined_Score'].max():.4f}")
    
    # ── محور ۳: ارزیابی Polynomial ───────────────────────────
    report.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    report.append("│  📈 محور ۳: ارزیابی رگرسیون چندجمله‌ای                           │")
    report.append("└─────────────────────────────────────────────────────────────────────┘")
    
    poly_models = [m for m in master_df['Model Name'] if m.startswith('Polynomial')]
    if poly_models:
        poly_model = poly_models[0]
        poly_r2 = master_df[master_df['Model Name'] == poly_model]['Test R²'].values[0]
        poly_train_r2 = master_df[master_df['Model Name'] == poly_model]['Train R²'].values[0]
        poly_gap = poly_train_r2 - poly_r2
        
        report.append(f"\n   • مدل: {poly_model}")
        report.append(f"   • Test R²: {poly_r2:.4f}")
        report.append(f"   • Train R²: {poly_train_r2:.4f}")
        report.append(f"   • R² Gap: {poly_gap:.4f}")
        
        if poly_gap > 0.1:
            report.append(f"\n   ⚠️ نتیجه: Polynomial دچار Overfitting شده است")
            report.append(f"      افزایش پیچیدگی توانی ارزش از دست رفتن تعمیم‌پذیری را نداشت")
        elif poly_r2 > best_balanced_r2:
            report.append(f"\n   ✅ نتیجه: Polynomial بهترین عملکرد را داشت")
            report.append(f"      افزایش پیچیدگی توانی ارزشمند بود")
        else:
            report.append(f"\n   ⚠️ نتیجه: Polynomial بهتر از Multivariate نیست")
            report.append(f"      پیچیدگی اضافی سودی نداشت")
    else:
        report.append(f"\n   ℹ️ مدل Polynomial در دسترس نیست")
    
    # ── محور ۴: توصیه نهایی مهندسی ───────────────────────────
    report.append("\n┌─────────────────────────────────────────────────────────────────────┐")
    report.append("│  🎯 محور ۴: توصیه نهایی مهندسی                                    │")
    report.append("└─────────────────────────────────────────────────────────────────────┘")
    
    # انتخاب مدل برتر نهایی
    overall_best = best_models.get('Test R²', 'N/A')
    overall_best_r2 = master_df[master_df['Model Name'] == overall_best]['Test R²'].values[0] if overall_best != 'N/A' else 0
    
    report.append(f"\n   🏆 مدل برتر نهایی: {overall_best}")
    report.append(f"      • Test R²: {overall_best_r2:.4f}")
    
    # دلایل انتخاب
    report.append(f"\n   📌 دلایل انتخاب:")
    
    # دلیل ۱: بالاترین R²
    report.append(f"      ۱. بالاترین Test R² در بین تمام مدل‌ها")
    
    # دلیل ۲: پایداری
    if overall_best == best_models.get('Stability (CV MSE Std)', 'N/A'):
        report.append(f"      ۲. پایدارترین مدل (کمترین CV MSE Std)")
    
    # دلیل ۳: سرعت
    if overall_best == best_models.get('Training Time', 'N/A'):
        report.append(f"      ۳. سریع‌ترین مدل در آموزش")
    
    # دلیل ۴: تفسیرپذیری
    if 'Simple' in overall_best or 'Multivariate' in overall_best or 'Ridge' in overall_best:
        report.append(f"      ۴. سطح تفسیرپذیری بالا")
    
    # توصیه کاربردی
    report.append(f"\n   💡 توصیه کاربردی:")
    if 'Polynomial' in overall_best:
        report.append(f"      • برای پیش‌بینی دقیق استفاده کنید")
        report.append(f"      • مراقب Overfitting در داده‌های جدید باشید")
        report.append(f"      • تفسیرپذیری پایین‌تر است")
    elif 'Lasso' in overall_best or 'ElasticNet' in overall_best:
        report.append(f"      • برای پیش‌بینی و انتخاب ویژگی استفاده کنید")
        report.append(f"      • مدل پایدار و قابل تعمیم")
        report.append(f"      • تفسیرپذیری خوب")
    elif 'Ridge' in overall_best:
        report.append(f"      • برای شرایط هم‌خطی شدید استفاده کنید")
        report.append(f"      • مدل پایدار با تفسیرپذیری بالا")
    else:
        report.append(f"      • مدل ساده و قابل تفسیر")
        report.append(f"      • مناسب برای درک اولیه داده‌ها")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report), overall_best, overall_best_r2


# ══════════════════════════════════════════════════════════════
#  بخش ۶ | تابع اصلی فاز ۱۷
# ══════════════════════════════════════════════════════════════

def run_executive_summary(
    phase5_result: Phase5Result,
    phase15_result: Phase15Result,
    output_folder: str = 'phase17_executive_summary'
) -> Phase17Result:
    """
    اجرای کامل فاز ۱۷ — تحلیل خودکار و نتیجه‌گیری نهایی
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل models_results)
        phase15_result: نتیجه فاز ۱۵ (شامل master_df)
        output_folder: نام پوشه ذخیره گزارش
    
    بازگشت:
        Phase17Result: شامل خلاصه، جدول مفهومی و توصیه نهایی
    """
    print("\n" + "=" * 70)
    print("🏆  فاز ۱۷ — تحلیل خودکار و نتیجه‌گیری نهایی")
    print("=" * 70)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره گزارش: {save_dir}")
    
    master_df = phase15_result.master_df
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: خلاصه خودکار بهترین‌ها
    # ══════════════════════════════════════════════════════════
    best_models = _extract_best_models(master_df, phase5_result)
    _print_executive_summary(best_models)
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: جدول مقایسه مفهومی
    # ══════════════════════════════════════════════════════════
    conceptual_df = _build_conceptual_matrix()
    _print_conceptual_matrix(conceptual_df)
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: تحلیل متنی و توصیه نهایی
    # ══════════════════════════════════════════════════════════
    final_recommendation, overall_best, overall_best_r2 = _generate_final_recommendation(
        master_df, phase5_result, best_models
    )
    print(final_recommendation)
    
    # ── ذخیره گزارش در فایل متنی ────────────────────────────
    report_path = save_dir / 'executive_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("🏆  گزارش نهایی پروژه — Executive Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("بخش اول: خلاصه بهترین‌ها\n")
        f.write("-" * 70 + "\n")
        for key, value in best_models.items():
            f.write(f"  🥇 {key}: {value}\n")
        
        f.write("\n\nبخش دوم: جدول مقایسه مفهومی\n")
        f.write("-" * 70 + "\n")
        f.write(conceptual_df.to_string(index=False))
        
        f.write("\n\n\nبخش سوم: توصیه نهایی\n")
        f.write("-" * 70 + "\n")
        f.write(final_recommendation)
    
    print(f"\n   ✅ گزارش ذخیره شد: {report_path.name}")
    
    # ══════════════════════════════════════════════════════════
    #  ساخت نتیجه
    # ══════════════════════════════════════════════════════════
    result = Phase17Result(
        best_models=best_models,
        conceptual_matrix=conceptual_df,
        final_recommendation=final_recommendation,
        overall_best_model=overall_best,
        overall_best_r2=overall_best_r2
    )
    
    # ── پایان فاز ۱۷ ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅  فاز ۱۷ — تحلیل خودکار و نتیجه‌گیری نهایی کامل شد.")
    print(f"   • مدل برتر نهایی: {overall_best}")
    print(f"   • Test R²: {overall_best_r2:.4f}")
    print(f"   • گزارش ذخیره‌شده: {report_path.name}")
    print("=" * 70)
    
    return result