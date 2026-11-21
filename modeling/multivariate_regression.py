"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۸ — رگرسیون خطی چندمتغیره (Multivariate Linear Regression)║
║  Phase 8 — Multivariate Linear Regression                   ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    توسعه مدل خطی با به‌کارگیری هم‌زمان تمام ویژگی‌های موجود
    در دیتاست جهت بهبود توان پیش‌بینی نسبت به مدل تک‌متغیره (فاز ۷).

خروجی‌ها:
    • معیارهای Test: MSE, RMSE, MAE, R²
    • معیارهای Train: MSE, R²
    • نتایج CV: Mean, Std برای MSE و R²
    • جدول ضرایب مرتب‌شده بر اساس قدرمطلق
    • مقایسه با مدل Simple LR (فاز ۷)
    • تحلیل اولیه Overfitting
    • ۵ نمودار (۳ تایی + ۲ تایی)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LinearRegression

from modeling.config import C, MODEL_COLOR_MAP, TARGET_COLUMN
from modeling.data_splitter import Phase4Result
from modeling.feature_selector import Phase2Result
from modeling.cross_validator import Phase5Result
from modeling.evaluator import evaluate_model, evaluation_result_to_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۸
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase8Result:
    """نتایج فاز ۸ — رگرسیون خطی چندمتغیره"""
    
    # مدل و پارامترها
    model: LinearRegression
    coefficients: np.ndarray          # ضرایب تمام ویژگی‌ها
    intercept: float                  # عرض از مبدأ
    feature_names: list               # لیست نام ویژگی‌ها
    
    # ضرایب مرتب‌شده (DataFrame)
    coefficients_sorted: pd.DataFrame
    
    # نتایج ارزیابی
    evaluation_result: object         # EvaluationResult از فاز ۶
    
    # مقایسه با فاز ۷
    comparison_with_simple: dict      # مقایسه با Simple LR
    
    # تحلیل Overfitting
    overfitting_gap: float            # Train R² - Test R²
    
    # نام مدل
    model_name: str = "Multivariate LR"


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی رسم نمودار (محلی برای فاز ۸)
# ══════════════════════════════════════════════════════════════

def _plot_actual_vs_predicted(y_true, y_pred, model_name, ax):
    """
    نمودار چپ: Actual vs Predicted با خط y=x
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی‌شده
        model_name: نام مدل
        ax: محور نمودار
    """
    # رسم نقاط
    ax.scatter(
        y_true, y_pred,
        alpha=0.3, s=12,
        color=C[1],  # رنگ Multivariate LR
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط y=x (خط کمال)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot(
        [lo, hi], [lo, hi],
        'r--', linewidth=1.5,
        label='y = ŷ (perfect)'
    )
    
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title('Actual vs Predicted', fontweight='normal', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')


def _plot_residuals(y_true, y_pred, model_name, ax):
    """
    نمودار وسط: Residual Plot
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی‌شده
        model_name: نام مدل
        ax: محور نمودار
    """
    # محاسبه باقیمانده‌ها
    residuals = y_true - y_pred
    
    # رسم نقاط
    ax.scatter(
        y_pred, residuals,
        alpha=0.3, s=12,
        color=C[1],
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط افقی روی صفر
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Residual (y - ŷ)', fontsize=11)
    ax.set_title('Residual Plot', fontweight='normal', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')


def _plot_coefficient_bar_chart(coefficients, feature_names, ax):
    """
    نمودار راست: Coefficient Bar Chart
    
    پارامترها:
        coefficients: آرایه ضرایب
        feature_names: لیست نام ویژگی‌ها
        ax: محور نمودار
    """
    # مرتب‌سازی بر اساس قدرمطلق (از بزرگ به کوچک)
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefs = coefficients[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # تعیین رنگ‌ها: آبی برای مثبت، قرمز برای منفی
    colors = ['steelblue' if c >= 0 else 'salmon' for c in sorted_coefs]
    
    # رسم میله‌های افقی
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()  # بزرگترین در بالا
    
    # خط عمودی روی صفر
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Coefficient Value', fontsize=11)
    ax.set_title('Feature Coefficients\n(Blue=Positive, Red=Negative)', 
                 fontweight='normal', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')


def _plot_cv_boxplots(cv_mse_scores, cv_r2_scores, model_name, save_dir):
    """
    نمودار دوم: Boxplot نتایج CV (MSE و R²)
    
    پارامترها:
        cv_mse_scores: نمرات MSE هر فولد
        cv_r2_scores: نمرات R² هر فولد
        model_name: نام مدل
        save_dir: مسیر ذخیره
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # باکس‌پلات MSE
    axes[0].boxplot(
        cv_mse_scores,
        patch_artist=True,
        boxprops=dict(facecolor=C[1], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[0].set_title(f'{model_name} — CV MSE Distribution', 
                      fontweight='normal', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=11)
    axes[0].set_xlabel('5-Fold CV', fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # اضافه کردن مقادیر عددی
    axes[0].text(
        1.05, cv_mse_scores.mean(),
        f'Mean: {cv_mse_scores.mean():.4f}',
        va='center', fontsize=10, color='red'
    )
    
    # باکس‌پلات R²
    axes[1].boxplot(
        cv_r2_scores,
        patch_artist=True,
        boxprops=dict(facecolor=C[2], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_title(f'{model_name} — CV R² Distribution', 
                      fontweight='normal', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=11)
    axes[1].set_xlabel('5-Fold CV', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # اضافه کردن مقادیر عددی
    axes[1].text(
        1.05, cv_r2_scores.mean(),
        f'Mean: {cv_r2_scores.mean():.4f}',
        va='center', fontsize=10, color='red'
    )
    
    fig.suptitle(
        f'Cross-Validation Stability — {model_name}',
        fontsize=14, fontweight='normal', y=1.02
    )
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig08_cv_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۸
# ══════════════════════════════════════════════════════════════

def run_multivariate_regression(
    phase4_result: Phase4Result,
    phase2_result: Phase2Result,
    phase5_result: Phase5Result,
    output_folder: str = 'phase8_multivariate_regression_plots'
) -> Phase8Result:
    """
    اجرای کامل فاز ۸ — رگرسیون خطی چندمتغیره
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_train, X_test)
        phase2_result: نتیجه فاز ۲ (شامل feature_names)
        phase5_result: نتیجه فاز ۵ (شامل cv و دیکشنری‌ها)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase8Result: شامل مدل، ضرایب و نتایج ارزیابی
    """
    MODEL_NAME = "Multivariate LR"
    
    print("\n" + "=" * 65)
    print("📌  فاز ۸ — رگرسیون خطی چندمتغیره (Multivariate Linear Regression)")
    print("=" * 65)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره نمودارها: {save_dir}")
    
    # ── استخراج داده‌ها ──────────────────────────────────────
    X_train = phase4_result.X_train
    X_test = phase4_result.X_test
    y_train = phase4_result.y_train
    y_test = phase4_result.y_test
    feature_names = phase2_result.feature_names
    
    print(f"\n▶  تعداد ویژگی‌ها: {len(feature_names)}")
    print(f"   • شکل X_train: {X_train.shape}")
    print(f"   • شکل X_test : {X_test.shape}")
    print(f"   • لیست ویژگی‌ها: {feature_names}")
    
    # ── ساخت مدل ─────────────────────────────────────────────
    print(f"\n▶  ساخت مدل LinearRegression (Multivariate)...")
    model = LinearRegression()
    
    # ── ارزیابی با تابع یکپارچه فاز ۶ ────────────────────────
    print(f"\n▶  ارسال به تابع ارزیابی یکپارچه...")
    eval_result = evaluate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        phase5_result=phase5_result,
        model_name=MODEL_NAME,
        verbose=True
    )
    
    # ── استخراج ضرایب ────────────────────────────────────────
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"\n▶  عرض از مبدأ (Intercept): {intercept:.4f}")
    
    # ── ساخت جدول ضرایب مرتب‌شده ────────────────────────────
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    })
    
    # مرتب‌سازی بر اساس قدرمطلق (از بزرگ به کوچک)
    coef_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_sorted = coef_sorted[['Feature', 'Coefficient']].reset_index(drop=True)
    coef_sorted.index = coef_sorted.index + 1  # شروع از ۱
    
    print(f"\n▶  جدول ضرایب (مرتب‌شده بر اساس میزان تأثیرگذاری):")
    print("-" * 65)
    print(coef_sorted.to_string())
    print("-" * 65)
    
    # ── مقایسه با فاز ۷ (Simple LR) ─────────────────────────
    comparison = {}
    if "Simple LR" in phase5_result.models_results:
        simple_result = phase5_result.models_results["Simple LR"]
        
        comparison = {
            'simple_test_mse': simple_result['test_mse'],
            'simple_test_r2': simple_result['test_r2'],
            'simple_test_rmse': simple_result['test_rmse'],
            'simple_test_mae': simple_result['test_mae'],
            'multi_test_mse': eval_result.test_mse,
            'multi_test_r2': eval_result.test_r2,
            'multi_test_rmse': eval_result.test_rmse,
            'multi_test_mae': eval_result.test_mae,
            'mse_improvement': simple_result['test_mse'] - eval_result.test_mse,
            'r2_improvement': eval_result.test_r2 - simple_result['test_r2'],
            'rmse_improvement': simple_result['test_rmse'] - eval_result.test_rmse,
            'mae_improvement': simple_result['test_mae'] - eval_result.test_mae,
        }
        
        print(f"\n{'─' * 65}")
        print(f"📊  مقایسه با مدل Simple LR (فاز ۷):")
        print(f"{'─' * 65}")
        print(f"\n   {'معیار':<20} {'Simple LR':<15} {'Multivariate LR':<15} {'بهبود':<15}")
        print(f"   {'-'*65}")
        print(f"   {'Test MSE':<20} {comparison['simple_test_mse']:<15.4f} {comparison['multi_test_mse']:<15.4f} {comparison['mse_improvement']:<15.4f}")
        print(f"   {'Test RMSE':<20} {comparison['simple_test_rmse']:<15.4f} {comparison['multi_test_rmse']:<15.4f} {comparison['rmse_improvement']:<15.4f}")
        print(f"   {'Test MAE':<20} {comparison['simple_test_mae']:<15.4f} {comparison['multi_test_mae']:<15.4f} {comparison['mae_improvement']:<15.4f}")
        print(f"   {'Test R²':<20} {comparison['simple_test_r2']:<15.4f} {comparison['multi_test_r2']:<15.4f} {comparison['r2_improvement']:<15.4f}")
        
        # تحلیل بهبود
        if comparison['r2_improvement'] > 0:
            print(f"\n   ✅ افزودن متغیرها منجر به بهبود مدل شد (R² افزایش یافت).")
        else:
            print(f"\n   ⚠️ افزودن متغیرها منجر به بهبود مدل نشد (R² کاهش یافت).")
    else:
        print(f"\n   ⚠️ نتایج مدل Simple LR در دسترس نیست. مقایسه انجام نشد.")
    
    # ── تحلیل اولیه Overfitting ──────────────────────────────
    overfitting_gap = eval_result.train_r2 - eval_result.test_r2
    
    print(f"\n{'─' * 65}")
    print(f"🔍  تحلیل اولیه Overfitting:")
    print(f"{'─' * 65}")
    print(f"\n   • Train R² : {eval_result.train_r2:.4f}")
    print(f"   • Test R²  : {eval_result.test_r2:.4f}")
    print(f"   • R² Gap   : {overfitting_gap:.4f}  (Train R² - Test R²)")
    
    if overfitting_gap > 0.1:
        print(f"\n   ⚠️ شکاف بزرگ (>{0.1}) — نشانه‌های اولیه Overfitting مشاهده می‌شود.")
    elif overfitting_gap > 0.05:
        print(f"\n   ⚠️ شکاف متوسط — مدل کمی به داده‌های آموزش وابسته است.")
    else:
        print(f"\n   ✅ شکاف کوچک — مدل تعمیم‌پذیری خوبی دارد.")
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase8Result(
        model=model,
        coefficients=coefficients,
        intercept=intercept,
        feature_names=feature_names,
        coefficients_sorted=coef_sorted,
        evaluation_result=eval_result,
        comparison_with_simple=comparison,
        overfitting_gap=overfitting_gap,
        model_name=MODEL_NAME
    )
    
    # ── ذخیره در دیکشنری‌های فاز ۵ ──────────────────────────
    phase5_result.models_results[MODEL_NAME] = evaluation_result_to_dict(eval_result)
    phase5_result.trained_models[MODEL_NAME] = model
    
    print(f"\n   ✅ نتایج مدل '{MODEL_NAME}' در دیکشنری‌ها ذخیره شد.")
    
    # ── چاپ خلاصه معیارها ────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"📊  خلاصه معیارهای عملکردی:")
    print(f"{'─' * 65}")
    
    print(f"\n   📌 معیارهای Test:")
    print(f"      • MSE  : {eval_result.test_mse:.4f}")
    print(f"      • RMSE : {eval_result.test_rmse:.4f}")
    print(f"      • MAE  : {eval_result.test_mae:.4f}")
    print(f"      • R²   : {eval_result.test_r2:.4f}")
    
    print(f"\n   📌 معیارهای Train:")
    print(f"      • MSE  : {eval_result.train_mse:.4f}")
    print(f"      • R²   : {eval_result.train_r2:.4f}")
    
    print(f"\n   📌 نتایج CV (5-Fold):")
    print(f"      • MSE Mean: {eval_result.cv_mse_mean:.4f} ± {eval_result.cv_mse_std:.4f}")
    print(f"      • R² Mean : {eval_result.cv_r2_mean:.4f} ± {eval_result.cv_r2_std:.4f}")
    
    print(f"\n   📌 زمان:")
    print(f"      • Training Time  : {eval_result.train_time:.5f} ثانیه")
    print(f"      • Prediction Time: {eval_result.pred_time:.5f} ثانیه")
    
    # ══════════════════════════════════════════════════════════
    #  رسم نمودارها
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📈  رسم نمودارهای فاز ۸...")
    print(f"{'─' * 65}")
    
    # ── نمودار اول: پنل ۳ تایی ───────────────────────────────
    print(f"\n   ▶  رسم پنل ۳ تایی (AvP + Residuals + Coefficients)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # چپ: Actual vs Predicted
    _plot_actual_vs_predicted(
        y_test, eval_result.y_pred_test,
        MODEL_NAME,
        axes[0]
    )
    
    # وسط: Residual Plot
    _plot_residuals(
        y_test, eval_result.y_pred_test,
        MODEL_NAME,
        axes[1]
    )
    
    # راست: Coefficient Bar Chart
    _plot_coefficient_bar_chart(
        coefficients, feature_names,
        axes[2]
    )
    
    fig.suptitle(
        f"Phase 8 | Multivariate Linear Regression — All {len(feature_names)} Features",
        fontsize=14, fontweight='normal', y=1.02
    )
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig08_main_panel.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()
    
    # ── نمودار دوم: Boxplot CV ───────────────────────────────
    print(f"\n   ▶  رسم Boxplot نتایج CV...")
    
    _plot_cv_boxplots(
        eval_result.cv_mse_scores,
        eval_result.cv_r2_scores,
        MODEL_NAME,
        save_dir
    )
    
    # ── پایان فاز ۸ ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  فاز ۸ — رگرسیون خطی چندمتغیره کامل شد.")
    print("=" * 65)
    
    return result