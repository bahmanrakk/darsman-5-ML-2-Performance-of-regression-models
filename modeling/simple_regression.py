"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۷ — رگرسیون خطی تک‌متغیره (Simple Linear Regression)   ║
║  Phase 7 — Simple Linear Regression (Baseline Model)        ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    پیاده‌سازی ساده‌ترین مدل خطی ممکن (خط مستقیم) بر اساس بهترین
    ویژگی انتخابی از فاز ۳ به عنوان Baseline پروژه.

خروجی‌ها:
    • معادله خط: y = Slope × X + Intercept
    • معیارهای Test: MSE, RMSE, MAE, R²
    • معیارهای Train: MSE, R²
    • نتایج CV: Mean, Std برای MSE و R²
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
from modeling.feature_selector import Phase3Result
from modeling.cross_validator import Phase5Result
from modeling.evaluator import evaluate_model, evaluation_result_to_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۷
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase7Result:
    """نتایج فاز ۷ — رگرسیون خطی تک‌متغیره (Simple Linear Regression)"""
    
    # مدل و پارامترها
    model: LinearRegression
    slope: float                    # ضریب شیب
    intercept: float                # عرض از مبدأ
    feature_name: str               # نام ویژگی استفاده شده
    
    # معادله خط
    equation: str                   # معادله به صورت رشته
    
    # نتایج ارزیابی
    evaluation_result: object       # EvaluationResult از فاز ۶
    
    # نام مدل
    model_name: str = "Simple LR"


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی رسم نمودار (محلی برای فاز ۷)
# ══════════════════════════════════════════════════════════════

def _plot_scatter_with_line(X_test, y_test, model, feature_name, target_name, ax):
    """
    نمودار چپ: Scatter + Regression Line
    
    پارامترها:
        X_test: داده‌های آزمون تک‌ویژگی
        y_test: مقادیر واقعی هدف
        model: مدل آموزش‌دیده
        feature_name: نام ویژگی
        target_name: نام متغیر هدف
        ax: محور نمودار
    """
    # رسم نقاط پراکنده
    ax.scatter(
        X_test, y_test,
        alpha=0.4, s=15,
        color=C[0],
        label='Test Data',
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط رگرسیون
    x_range = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
    ax.plot(
        x_range, model.predict(x_range),
        color='red', linewidth=2,
        label='Regression Line'
    )
    
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel(target_name, fontsize=11)
    ax.set_title('Scatter + Regression Line', fontweight='normal', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')


def _plot_actual_vs_predicted(y_true, y_pred, model_name, ax):
    """
    نمودار وسط: Actual vs Predicted با خط y=x
    
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
        color=C[0],
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
    نمودار راست: Residual Plot
    
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
        color=C[0],
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط افقی روی صفر
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Residual (y - ŷ)', fontsize=11)
    ax.set_title('Residual Plot', fontweight='normal', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')


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
        boxprops=dict(facecolor=C[0], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[0].set_title(f'{model_name} — CV MSE Distribution', fontweight='normal', fontsize=12)
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
        boxprops=dict(facecolor=C[1], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_title(f'{model_name} — CV R² Distribution', fontweight='normal', fontsize=12)
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
    save_path = save_dir / 'fig07_cv_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۷
# ══════════════════════════════════════════════════════════════

def run_simple_regression(
    phase4_result: Phase4Result,
    phase3_result: Phase3Result,
    phase5_result: Phase5Result,
    output_folder: str = 'phase7_simple_regression_plots'
) -> Phase7Result:
    """
    اجرای کامل فاز ۷ — رگرسیون خطی تک‌متغیره (Simple Linear Regression)
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_uni_train, X_uni_test)
        phase3_result: نتیجه فاز ۳ (شامل best_feature_name)
        phase5_result: نتیجه فاز ۵ (شامل cv و دیکشنری‌ها)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase7Result: شامل مدل، پارامترها و نتایج ارزیابی
    """
    MODEL_NAME = "Simple LR"
    
    print("\n" + "=" * 65)
    print("📌  فاز ۷ — رگرسیون خطی تک‌متغیره (Simple Linear Regression)")
    print("=" * 65)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره نمودارها: {save_dir}")
    
    # ── استخراج داده‌ها ──────────────────────────────────────
    X_uni_train = phase4_result.X_uni_train
    X_uni_test = phase4_result.X_uni_test
    y_train = phase4_result.y_train
    y_test = phase4_result.y_test
    feature_name = phase3_result.best_feature_name
    
    print(f"\n▶  ویژگی انتخابی: '{feature_name}'")
    print(f"   • شکل X_uni_train: {X_uni_train.shape}")
    print(f"   • شکل X_uni_test : {X_uni_test.shape}")
    
    # ── ساخت مدل ─────────────────────────────────────────────
    print(f"\n▶  ساخت مدل LinearRegression (Simple)...")
    model = LinearRegression()
    
    # ── ارزیابی با تابع یکپارچه فاز ۶ ────────────────────────
    print(f"\n▶  ارسال به تابع ارزیابی یکپارچه...")
    eval_result = evaluate_model(
        model=model,
        X_train=X_uni_train,
        y_train=y_train,
        X_test=X_uni_test,
        y_test=y_test,
        phase5_result=phase5_result,
        model_name=MODEL_NAME,
        verbose=True
    )
    
    # ── استخراج پارامترهای هندسی خط ──────────────────────────
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {slope:.4f} × {feature_name} + {intercept:.4f}"
    
    print(f"\n▶  پارامترهای هندسی خط:")
    print(f"   • ضریب شیب (Slope)       : {slope:.4f}")
    print(f"   • عرض از مبدأ (Intercept): {intercept:.4f}")
    print(f"   • معادله خط: {equation}")
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase7Result(
        model=model,
        slope=slope,
        intercept=intercept,
        feature_name=feature_name,
        equation=equation,
        evaluation_result=eval_result,
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
    print(f"📈  رسم نمودارهای فاز ۷...")
    print(f"{'─' * 65}")
    
    # ── نمودار اول: پنل ۳ تایی ───────────────────────────────
    print(f"\n   ▶  رسم پنل ۳ تایی (Scatter + AvP + Residuals)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # چپ: Scatter + Regression Line
    _plot_scatter_with_line(
        X_uni_test, y_test, model,
        feature_name, TARGET_COLUMN,
        axes[0]
    )
    
    # وسط: Actual vs Predicted
    _plot_actual_vs_predicted(
        y_test, eval_result.y_pred_test,
        MODEL_NAME,
        axes[1]
    )
    
    # راست: Residual Plot
    _plot_residuals(
        y_test, eval_result.y_pred_test,
        MODEL_NAME,
        axes[2]
    )
    
    fig.suptitle(
        f"Phase 7 | Simple Linear Regression — '{feature_name}' vs Target",
        fontsize=14, fontweight='normal', y=1.02
    )
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig07_main_panel.png'
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
    
    # ── پایان فاز ۷ ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  فاز ۷ — رگرسیون خطی تک‌متغیره (Simple Linear Regression) کامل شد.")
    print("=" * 65)
    
    return result