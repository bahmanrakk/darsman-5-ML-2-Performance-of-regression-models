"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۰ — رگرسیون ریج (Ridge Regression - L₂ Regularization)║
║  Phase 10 — Ridge Regression with GridSearchCV              ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    اعمال جریمه مرتبه دوم (L₂) روی مجذور بزرگی ضرایب جهت کنترل
    پیچیدگی، کاهش واریانس مدل و مهار Overfitting.

کارهای کلیدی:
    • تعریف فضای جستجو: ۶۰ عدد لگاریتمی از 0.0001 تا 10000
    • Pipeline استاندارد: StandardScaler → Ridge
    • GridSearchCV با KFold فاز ۵
    • استخراج بهترین alpha و ارزیابی مدل نهایی
    • رسم نمودارهای Regularization Path و MSE vs Alpha

خروجی‌ها:
    • بهترین مقدار alpha
    • جدول معیارهای ارزیابی کامل
    • جدول ضرایب ویژگی‌ها
    • ۴ نمودار (MSE vs Alpha, Ridge Path, AvP, Residual, Boxplot CV)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from modeling.config import C, MODEL_COLOR_MAP, TARGET_COLUMN
from modeling.data_splitter import Phase4Result
from modeling.feature_selector import Phase2Result
from modeling.cross_validator import Phase5Result
from modeling.evaluator import evaluate_model, evaluation_result_to_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۰
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase10Result:
    """نتایج فاز ۱۰ — رگرسیون ریج"""
    
    # پارامترهای بهینه
    best_alpha: float
    best_pipeline: Pipeline
    
    # ضرایب
    coefficients: np.ndarray
    feature_names: list
    coefficients_df: pd.DataFrame
    
    # نتایج GridSearchCV
    grid_search_results: Dict[str, Any]
    cv_results_df: pd.DataFrame
    
    # نتایج ارزیابی مدل نهایی
    evaluation_result: object
    
    # نام مدل
    model_name: str = "Ridge"


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی رسم نمودار
# ══════════════════════════════════════════════════════════════

def _plot_mse_vs_alpha(alphas, cv_results, best_alpha, save_dir):
    """
    نمودار چپ: MSE vs Alpha با نوار اطمینان
    
    پارامترها:
        alphas: آرایه مقادیر alpha
        cv_results: نتایج GridSearchCV
        best_alpha: بهترین alpha
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # استخراج میانگین و انحراف معیار CV MSE
    cv_mse_mean = -cv_results['mean_test_score']
    cv_mse_std = cv_results['std_test_score']
    
    # رسم خط اصلی
    ax.semilogx(alphas, cv_mse_mean, color=C[3], linewidth=2, 
                label='CV MSE (Mean)')
    
    # رسم نوار اطمینان (Mean ± Std)
    ax.fill_between(
        alphas,
        cv_mse_mean - cv_mse_std,
        cv_mse_mean + cv_mse_std,
        alpha=0.3, color=C[3],
        label='CV MSE (Mean ± Std)'
    )
    
    # خط عمودی روی بهترین alpha
    ax.axvline(best_alpha, color='red', linestyle='--', linewidth=2,
               label=f'Best α = {best_alpha:.4f}')
    
    ax.set_xlabel('Alpha (log scale)', fontsize=12)
    ax.set_ylabel('Cross-Validation MSE', fontsize=12)
    ax.set_title('Ridge Regression — CV MSE vs Alpha', 
                 fontweight='normal', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig10a_mse_vs_alpha.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


def _plot_ridge_path(alphas, coefficients_path, feature_names, best_alpha, save_dir):
    """
    نمودار راست: Ridge Regularization Path
    
    پارامترها:
        alphas: آرایه مقادیر alpha
        coefficients_path: ماتریس ضرایب (n_alphas × n_features)
        feature_names: لیست نام ویژگی‌ها
        best_alpha: بهترین alpha
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # رسم مسیر ضرایب برای هر ویژگی
    for i, feature_name in enumerate(feature_names):
        ax.semilogx(
            alphas, coefficients_path[:, i],
            linewidth=1.5, label=feature_name
        )
    
    # خط عمودی روی بهترین alpha
    ax.axvline(best_alpha, color='red', linestyle='--', linewidth=2,
               label=f'Best α = {best_alpha:.4f}')
    
    # خط افقی روی صفر
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    
    ax.set_xlabel('Alpha (log scale)', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Ridge Regularization Path\n(Coefficients → 0 but never exactly 0)', 
                 fontweight='normal', fontsize=13)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig10b_ridge_path.png'
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
        color=C[3],
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
    save_path = save_dir / 'fig10c_actual_vs_predicted.png'
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
        color=C[3],
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
    save_path = save_dir / 'fig10d_residuals.png'
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
        boxprops=dict(facecolor=C[3], alpha=0.7),
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
        boxprops=dict(facecolor=C[4], alpha=0.7),
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
    save_path = save_dir / 'fig10e_cv_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۱۰
# ══════════════════════════════════════════════════════════════

def run_ridge_regression(
    phase4_result: Phase4Result,
    phase2_result: Phase2Result,
    phase5_result: Phase5Result,
    output_folder: str = 'phase10_ridge_regression_plots'
) -> Phase10Result:
    """
    اجرای کامل فاز ۱۰ — رگرسیون ریج
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_train, X_test)
        phase2_result: نتیجه فاز ۲ (شامل feature_names)
        phase5_result: نتیجه فاز ۵ (شامل cv و دیکشنری‌ها)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase10Result: شامل بهترین alpha، ضرایب و نتایج ارزیابی
    """
    MODEL_NAME = "Ridge"
    
    print("\n" + "=" * 65)
    print("📌  فاز ۱۰ — رگرسیون ریج (Ridge Regression - L₂ Regularization)")
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
    cv = phase5_result.cv
    
    print(f"\n▶  تنظیمات:")
    print(f"   • تعداد ویژگی‌ها: {len(feature_names)}")
    print(f"   • Pipeline: StandardScaler → Ridge")
    print(f"   • معیار GridSearch: neg_mean_squared_error")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: تعریف فضای جستجو و GridSearchCV
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🔍  بخش اول: جستجوی بهینه alpha")
    print(f"{'─' * 65}")
    
    # ── تعریف بازه alpha (۶۰ عدد لگاریتمی) ───────────────────
    alphas = np.logspace(-4, 4, 60)  # از 0.0001 تا 10000
    
    print(f"\n▶  فضای جستجو:")
    print(f"   • تعداد مقادیر alpha: {len(alphas)}")
    print(f"   • محدوده: {alphas[0]:.4f} تا {alphas[-1]:.4f}")
    print(f"   • مقیاس: لگاریتمی")
    
    # ── ساخت Pipeline استاندارد ──────────────────────────────
    print(f"\n▶  ساخت Pipeline: StandardScaler → Ridge...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    
    # ── تعریف گرید پارامترها ─────────────────────────────────
    param_grid = {
        'ridge__alpha': alphas
    }
    
    # ── اجرای GridSearchCV ───────────────────────────────────
    print(f"\n▶  اجرای GridSearchCV با {cv.n_splits}-Fold CV...")
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # ── استخراج بهترین alpha ─────────────────────────────────
    best_alpha = grid_search.best_params_['ridge__alpha']
    best_pipeline = grid_search.best_estimator_
    
    print(f"\n   ✅ بهترین alpha: {best_alpha:.6f}")
    print(f"   • بهترین CV MSE: {-grid_search.best_score_:.4f}")
    
    # ── استخراج نتایج GridSearchCV ───────────────────────────
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['mean_test_mse'] = -cv_results_df['mean_test_score']
    cv_results_df['std_test_mse'] = cv_results_df['std_test_score']
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: محاسبه مسیر ضرایب (Ridge Path)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش دوم: محاسبه مسیر ضرایب")
    print(f"{'─' * 65}")
    
    print(f"\n▶  محاسبه ضرایب برای تمام مقادیر alpha...")
    
    coefficients_path = []
    for alpha in alphas:
        temp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        temp_pipeline.fit(X_train, y_train)
        coefficients_path.append(temp_pipeline.named_steps['ridge'].coef_)
    
    coefficients_path = np.array(coefficients_path)
    
    print(f"   • شکل ماتریس ضرایب: {coefficients_path.shape}")
    print(f"   • تعداد ویژگی‌ها: {coefficients_path.shape[1]}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: ارزیابی مدل نهایی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🎯  بخش سوم: ارزیابی مدل نهایی")
    print(f"{'─' * 65}")
    
    # ── استخراج ضرایب نهایی ──────────────────────────────────
    coefficients = best_pipeline.named_steps['ridge'].coef_
    intercept = best_pipeline.named_steps['ridge'].intercept_
    
    # ── ساخت DataFrame ضرایب ─────────────────────────────────
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    })
    coef_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_sorted = coef_sorted[['Feature', 'Coefficient']].reset_index(drop=True)
    coef_sorted.index = coef_sorted.index + 1
    
    print(f"\n▶  جدول ضرایب (مرتب‌شده بر اساس قدرمطلق):")
    print("-" * 65)
    print(coef_sorted.to_string())
    print("-" * 65)
    
    # ── تحلیل ضرایب ──────────────────────────────────────────
    print(f"\n▶  تحلیل ضرایب:")
    print(f"   • حداکثر ضریب: {coefficients.max():.4f}")
    print(f"   • حداقل ضریب: {coefficients.min():.4f}")
    print(f"   • میانگین قدرمطلق: {np.abs(coefficients).mean():.4f}")
    print(f"   • تعداد ضرایب مثبت: {(coefficients > 0).sum()}")
    print(f"   • تعداد ضرایب منفی: {(coefficients < 0).sum()}")
    
    # ── ارزیابی با تابع یکپارچه فاز ۶ ────────────────────────
    print(f"\n▶  ارسال به تابع ارزیابی یکپارچه...")
    
    eval_result = evaluate_model(
        model=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        phase5_result=phase5_result,
        model_name=MODEL_NAME,
        verbose=True
    )
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase10Result(
        best_alpha=best_alpha,
        best_pipeline=best_pipeline,
        coefficients=coefficients,
        feature_names=feature_names,
        coefficients_df=coef_sorted,
        grid_search_results={
            'best_alpha': best_alpha,
            'best_score': -grid_search.best_score_,
            'n_alphas_tested': len(alphas)
        },
        cv_results_df=cv_results_df,
        evaluation_result=eval_result,
        model_name=MODEL_NAME
    )
    
    # ── ذخیره در دیکشنری‌های فاز ۵ ──────────────────────────
    phase5_result.models_results[MODEL_NAME] = evaluation_result_to_dict(eval_result)
    phase5_result.trained_models[MODEL_NAME] = best_pipeline
    
    # ذخیره بهترین alpha برای استفاده در فازهای بعدی
    phase5_result.models_results[MODEL_NAME]['best_alpha'] = best_alpha
    
    print(f"\n   ✅ نتایج مدل '{MODEL_NAME}' در دیکشنری‌ها ذخیره شد.")
    
    # ── چاپ خلاصه معیارها ────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"📊  خلاصه معیارهای عملکردی:")
    print(f"{'─' * 65}")
    
    print(f"\n   📌 بهترین alpha: {best_alpha:.6f}")
    
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
        print(f"\n   ⚠️ شکاف بزرگ — مدل هنوز دچار Overfitting است.")
    elif overfitting_gap > 0.05:
        print(f"\n   ⚠️ شکاف متوسط — Regularization تا حدی مؤثر بوده است.")
    else:
        print(f"\n   ✅ شکاف کوچک — Regularization مؤثر بوده است.")
    
    # ══════════════════════════════════════════════════════════
    #  رسم نمودارها
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📈  رسم نمودارهای فاز ۱۰...")
    print(f"{'─' * 65}")
    
    # ── نمودار اول: پنل ۲ تایی ───────────────────────────────
    print(f"\n   ▶  رسم پنل ۲ تایی (MSE vs Alpha + Ridge Path)...")
    
    # چپ: MSE vs Alpha
    _plot_mse_vs_alpha(alphas, grid_search.cv_results_, best_alpha, save_dir)
    
    # راست: Ridge Regularization Path
    _plot_ridge_path(alphas, coefficients_path, feature_names, best_alpha, save_dir)
    
    # ── نمودار دوم: AvP + Residuals ──────────────────────────
    print(f"\n   ▶  رسم Actual vs Predicted...")
    _plot_actual_vs_predicted(y_test, eval_result.y_pred_test, MODEL_NAME, save_dir)
    
    print(f"\n   ▶  رسم Residual Plot...")
    _plot_residuals(y_test, eval_result.y_pred_test, MODEL_NAME, save_dir)
    
    # ── نمودار سوم: Boxplot CV ───────────────────────────────
    print(f"\n   ▶  رسم Boxplot نتایج CV...")
    _plot_cv_boxplots(
        eval_result.cv_mse_scores,
        eval_result.cv_r2_scores,
        MODEL_NAME,
        save_dir
    )
    
    # ── پایان فاز ۱۰ ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"✅  فاز ۱۰ — رگرسیون ریج (α = {best_alpha:.6f}) کامل شد.")
    print("=" * 65)
    
    return result