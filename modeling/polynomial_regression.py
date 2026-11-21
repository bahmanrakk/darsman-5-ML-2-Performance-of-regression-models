"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۹ — رگرسیون چندجمله‌ای با انتخاب بهترین درجه           ║
║  Phase 9 — Polynomial Regression with Optimal Degree        ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    عبور از محدودیت روابط خطی و مدل‌سازی الگوهای انحنا‌دار و پیچیده
    با تولید ویژگی‌های توانی، ضمن کنترل دقیق پیچیدگی مدل برای
    جلوگیری از Overfitting.

بخش اول:
    آزمون سیستماتیک درجات ۱ تا POLY_MAX_DEG با Pipeline استاندارد
    (PolynomialFeatures → StandardScaler → LinearRegression)

بخش دوم:
    انتخاب بهترین درجه بر اساس Minimum CV MSE و اجرای مدل نهایی

خروجی‌ها:
    • جدول مقایسه درجات (Degree, Num Features, Train MSE, Test MSE, CV MSE)
    • اعلام بهترین درجه با دلیل ریاضی
    • گزارش کامل معیارهای مدل نهایی
    • ۴ نمودار (Learning Curve, AvP, Residual, Boxplot CV)
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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from modeling.config import C, MODEL_COLOR_MAP, TARGET_COLUMN, POLY_MAX_DEG
from modeling.data_splitter import Phase4Result
from modeling.cross_validator import Phase5Result
from modeling.evaluator import evaluate_model, evaluation_result_to_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۹
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase9Result:
    """نتایج فاز ۹ — رگرسیون چندجمله‌ای"""
    
    # اطلاعات درجات آزمایش‌شده
    degrees_tested: List[int]
    degree_results: pd.DataFrame      # جدول مقایسه درجات
    
    # بهترین درجه
    best_degree: int
    best_pipeline: Pipeline
    
    # نتایج ارزیابی مدل نهایی
    evaluation_result: object
    
    # تعداد ویژگی‌های مدل نهایی
    n_features_final: int
    
    # نام مدل
    model_name: str


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی رسم نمودار
# ══════════════════════════════════════════════════════════════

def _plot_learning_curve(degrees, train_mse, test_mse, cv_mse, best_degree, save_dir):
    """
    نمودار Learning Curve: MSE vs Degree
    
    پارامترها:
        degrees: لیست درجات آزمایش‌شده
        train_mse: خطای Train برای هر درجه
        test_mse: خطای Test برای هر درجه
        cv_mse: خطای CV برای هر درجه
        best_degree: درجه بهینه
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # رسم سه خط
    ax.plot(degrees, train_mse, 'o-', color=C[0], linewidth=2, 
            markersize=8, label='Train MSE')
    ax.plot(degrees, test_mse, 's-', color=C[1], linewidth=2, 
            markersize=8, label='Test MSE')
    ax.plot(degrees, cv_mse, '^-', color=C[2], linewidth=2, 
            markersize=8, label='CV MSE')
    
    # خط عمودی روی درجه بهینه
    ax.axvline(best_degree, color='red', linestyle='--', linewidth=2,
               label=f'Best Degree = {best_degree}')
    
    # تنظیمات محور
    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('Learning Curve — MSE vs Polynomial Degree', 
                 fontweight='normal', fontsize=13)
    ax.set_xticks(degrees)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # اضافه کردن برچسب‌های عددی روی نقاط
    for i, deg in enumerate(degrees):
        ax.annotate(f'{cv_mse[i]:.2e}', 
                   (deg, cv_mse[i]),
                   textcoords="offset points", 
                   xytext=(0, 10),
                   ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig09a_learning_curve.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


def _plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir):
    """
    نمودار Actual vs Predicted
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی‌شده
        model_name: نام مدل
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # رسم نقاط
    ax.scatter(
        y_true, y_pred,
        alpha=0.3, s=12,
        color=C[2],
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط y=x
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot(
        [lo, hi], [lo, hi],
        'r--', linewidth=2,
        label='y = ŷ (perfect)'
    )
    
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'Actual vs Predicted — {model_name}', 
                 fontweight='normal', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig09b_actual_vs_predicted.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


def _plot_residuals(y_true, y_pred, model_name, save_dir):
    """
    نمودار Residual Plot
    
    پارامترها:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی‌شده
        model_name: نام مدل
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # محاسبه باقیمانده‌ها
    residuals = y_true - y_pred
    
    # رسم نقاط
    ax.scatter(
        y_pred, residuals,
        alpha=0.3, s=12,
        color=C[2],
        edgecolors='white',
        linewidth=0.3
    )
    
    # رسم خط افقی روی صفر
    ax.axhline(0, color='red', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Residual (y - ŷ)', fontsize=11)
    ax.set_title(f'Residual Plot — {model_name}', 
                 fontweight='normal', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig09c_residuals.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


def _plot_cv_boxplots(cv_mse_scores, cv_r2_scores, model_name, save_dir):
    """
    نمودار Boxplot نتایج CV
    
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
        boxprops=dict(facecolor=C[2], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[0].set_title(f'{model_name} — CV MSE Distribution', 
                      fontweight='normal', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=11)
    axes[0].set_xlabel('5-Fold CV', fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].text(
        1.05, cv_mse_scores.mean(),
        f'Mean: {cv_mse_scores.mean():.4f}',
        va='center', fontsize=10, color='red'
    )
    
    # باکس‌پلات R²
    axes[1].boxplot(
        cv_r2_scores,
        patch_artist=True,
        boxprops=dict(facecolor=C[3], alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_title(f'{model_name} — CV R² Distribution', 
                      fontweight='normal', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=11)
    axes[1].set_xlabel('5-Fold CV', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
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
    save_path = save_dir / 'fig09d_cv_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۹
# ══════════════════════════════════════════════════════════════

def run_polynomial_regression(
    phase4_result: Phase4Result,
    phase5_result: Phase5Result,
    max_degree: int = POLY_MAX_DEG,
    output_folder: str = 'phase9_polynomial_regression_plots'
) -> Phase9Result:
    """
    اجرای کامل فاز ۹ — رگرسیون چندجمله‌ای
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_train, X_test)
        phase5_result: نتیجه فاز ۵ (شامل cv و دیکشنری‌ها)
        max_degree: حداکثر درجه برای آزمایش (پیش‌فرض: از config.py)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase9Result: شامل بهترین درجه، پایپ‌لاین و نتایج ارزیابی
    """
    print("\n" + "=" * 65)
    print("📌  فاز ۹ — رگرسیون چندجمله‌ای (Polynomial Regression)")
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
    cv = phase5_result.cv
    
    print(f"\n▶  تنظیمات آزمایش:")
    print(f"   • حداکثر درجه: {max_degree}")
    print(f"   • درجات آزمایش‌شده: ۱ تا {max_degree}")
    print(f"   • Pipeline: PolynomialFeatures → StandardScaler → LinearRegression")
    print(f"   • معیار انتخاب بهترین درجه: Minimum CV MSE")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: آزمون درجات ۱ تا max_degree
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🔬  بخش اول: آزمون سیستماتیک درجات")
    print(f"{'─' * 65}")
    
    degrees = list(range(1, max_degree + 1))
    train_mse_list = []
    test_mse_list = []
    cv_mse_list = []
    n_features_list = []
    
    for degree in degrees:
        print(f"\n   ▶  آزمایش درجه {degree}...")
        
        # ساخت Pipeline استاندارد
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        
        # آموزش مدل
        pipeline.fit(X_train, y_train)
        
        # محاسبه تعداد ویژگی‌های تولیدشده
        n_features = pipeline.named_steps['poly'].n_output_features_
        n_features_list.append(n_features)
        
        # محاسبه Train MSE
        y_pred_train = pipeline.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_mse_list.append(train_mse)
        
        # محاسبه Test MSE
        y_pred_test = pipeline.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mse_list.append(test_mse)
        
        # محاسبه CV MSE
        cv_neg_mse = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_mse = -cv_neg_mse.mean()
        cv_mse_list.append(cv_mse)
        
        print(f"      • تعداد ویژگی‌ها: {n_features}")
        print(f"      • Train MSE: {train_mse:.4e}")
        print(f"      • Test MSE : {test_mse:.4e}")
        print(f"      • CV MSE   : {cv_mse:.4e}")
    
    # ── ساخت جدول مقایسه درجات ──────────────────────────────
    degree_results = pd.DataFrame({
        'Degree': degrees,
        'Num Features': n_features_list,
        'Train MSE': train_mse_list,
        'Test MSE': test_mse_list,
        'CV MSE': cv_mse_list
    })
    
    print(f"\n{'─' * 65}")
    print(f"📊  جدول مقایسه درجات:")
    print(f"{'─' * 65}")
    print(degree_results.to_string(index=False))
    print(f"{'─' * 65}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: انتخاب بهترین درجه و اجرای مدل نهایی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🎯  بخش دوم: انتخاب بهترین درجه")
    print(f"{'─' * 65}")
    
    # انتخاب بهترین درجه بر اساس Minimum CV MSE
    best_degree_idx = np.argmin(cv_mse_list)
    best_degree = degrees[best_degree_idx]
    best_cv_mse = cv_mse_list[best_degree_idx]
    
    print(f"\n   ✅ بهترین درجه: {best_degree}")
    print(f"   • دلیل: Minimum CV MSE = {best_cv_mse:.4e}")
    print(f"   • تعداد ویژگی‌های تولیدشده: {n_features_list[best_degree_idx]}")
    
    # ── ساخت پایپ‌لاین نهایی ─────────────────────────────────
    print(f"\n▶  ساخت پایپ‌لاین نهایی با درجه {best_degree}...")
    
    best_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    
    # نام مدل برای ذخیره
    model_name = f'Polynomial (deg={best_degree})'
    
    # اضافه کردن به MODEL_COLOR_MAP
    MODEL_COLOR_MAP[model_name] = C[2]
    
    # ── ارزیابی با تابع یکپارچه فاز ۶ ────────────────────────
    print(f"\n▶  ارسال به تابع ارزیابی یکپارچه...")
    eval_result = evaluate_model(
        model=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        phase5_result=phase5_result,
        model_name=model_name,
        verbose=True
    )
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase9Result(
        degrees_tested=degrees,
        degree_results=degree_results,
        best_degree=best_degree,
        best_pipeline=best_pipeline,
        evaluation_result=eval_result,
        n_features_final=n_features_list[best_degree_idx],
        model_name=model_name
    )
    
    # ── ذخیره در دیکشنری‌های فاز ۵ ──────────────────────────
    phase5_result.models_results[model_name] = evaluation_result_to_dict(eval_result)
    phase5_result.trained_models[model_name] = best_pipeline
    
    print(f"\n   ✅ نتایج مدل '{model_name}' در دیکشنری‌ها ذخیره شد.")
    
    # ── چاپ خلاصه معیارها ────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"📊  خلاصه معیارهای عملکردی مدل نهایی:")
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
    
    # ── تحلیل Overfitting ────────────────────────────────────
    overfitting_gap = eval_result.train_r2 - eval_result.test_r2
    
    print(f"\n{'─' * 65}")
    print(f"🔍  تحلیل Overfitting:")
    print(f"{'─' * 65}")
    print(f"\n   • Train R² : {eval_result.train_r2:.4f}")
    print(f"   • Test R²  : {eval_result.test_r2:.4f}")
    print(f"   • R² Gap   : {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.1:
        print(f"\n   ⚠️ شکاف بزرگ — مدل چندجمله‌ای دچار Overfitting شده است.")
    elif overfitting_gap > 0.05:
        print(f"\n   ⚠️ شکاف متوسط — مدل کمی به داده‌های آموزش وابسته است.")
    else:
        print(f"\n   ✅ شکاف کوچک — مدل تعمیم‌پذیری خوبی دارد.")
    
    # ══════════════════════════════════════════════════════════
    #  رسم نمودارها
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📈  رسم نمودارهای فاز ۹...")
    print(f"{'─' * 65}")
    
    # ── نمودار اول: Learning Curve ───────────────────────────
    print(f"\n   ▶  رسم Learning Curve (MSE vs Degree)...")
    
    _plot_learning_curve(
        degrees, train_mse_list, test_mse_list, cv_mse_list,
        best_degree, save_dir
    )
    
    # ── نمودار دوم: Actual vs Predicted ──────────────────────
    print(f"\n   ▶  رسم Actual vs Predicted...")
    
    _plot_actual_vs_predicted(
        y_test, eval_result.y_pred_test,
        model_name, save_dir
    )
    
    # ── نمودار سوم: Residual Plot ────────────────────────────
    print(f"\n   ▶  رسم Residual Plot...")
    
    _plot_residuals(
        y_test, eval_result.y_pred_test,
        model_name, save_dir
    )
    
    # ── نمودار چهارم: Boxplot CV ─────────────────────────────
    print(f"\n   ▶  رسم Boxplot نتایج CV...")
    
    _plot_cv_boxplots(
        eval_result.cv_mse_scores,
        eval_result.cv_r2_scores,
        model_name,
        save_dir
    )
    
    # ── پایان فاز ۹ ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"✅  فاز ۹ — رگرسیون چندجمله‌ای (درجه بهینه: {best_degree}) کامل شد.")
    print("=" * 65)
    
    return result