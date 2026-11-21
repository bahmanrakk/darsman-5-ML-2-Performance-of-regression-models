"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۲ — رگرسیون الاستیک نت (ElasticNet - L₁ + L₂)        ║
║  Phase 12 — ElasticNet Regression with 2D Grid Search       ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    ترکیب هم‌زمان جریمه‌های L₁ و L₂ جهت بهره‌مندی توامان از:
    • پایداری مدل ریج در شرایط هم‌خطی شدید
    • قدرت انتخاب ویژگی مدل لاسو

کارهای کلیدی:
    • گرید دوبعدی: ۲۰ مقدار alpha × ۷ مقدار l1_ratio = ۱۴۰ ترکیب
    • Pipeline استاندارد: StandardScaler → ElasticNet (max_iter=10000)
    • GridSearchCV با KFold فاز ۵
    • استخراج بهترین (alpha, l1_ratio) و ارزیابی مدل نهایی
    • شناسایی ویژگی‌های حذف‌شده (ضریب صفر)
    • مقایسه با Ridge و Lasso
    • رسم Heatmap دوبعدی و نمودار ضرایب

خروجی‌ها:
    • بهترین alpha و l1_ratio
    • جدول معیارهای ارزیابی کامل
    • لیست ویژگی‌های انتخاب‌شده و حذف‌شده
    • مقایسه با Ridge و Lasso
    • ۴ نمودار (Heatmap, Bar Chart, AvP, Boxplot CV)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from modeling.config import C, MODEL_COLOR_MAP, TARGET_COLUMN
from modeling.data_splitter import Phase4Result
from modeling.feature_selector import Phase2Result
from modeling.cross_validator import Phase5Result
from modeling.evaluator import evaluate_model, evaluation_result_to_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۲
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase12Result:
    """نتایج فاز ۱۲ — رگرسیون الاستیک نت"""
    
    # پارامترهای بهینه
    best_alpha: float
    best_l1_ratio: float
    best_pipeline: Pipeline
    
    # ضرایب
    coefficients: np.ndarray
    feature_names: list
    coefficients_df: pd.DataFrame
    
    # ویژگی‌های انتخاب‌شده و حذف‌شده
    selected_features: List[str]
    zeroed_features: List[str]
    n_selected: int
    n_zeroed: int
    
    # نتایج GridSearchCV
    grid_search_results: Dict[str, Any]
    cv_results_df: pd.DataFrame
    heat_matrix: np.ndarray           # ماتریس ۲ بعدی برای heatmap
    unique_alphas: np.ndarray
    unique_l1_ratios: np.ndarray
    
    # نتایج ارزیابی مدل نهایی
    evaluation_result: object
    
    # مقایسه با Ridge و Lasso
    comparison_with_ridge: Dict[str, Any]
    comparison_with_lasso: Dict[str, Any]
    
    # نام مدل
    model_name: str = "ElasticNet"


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی رسم نمودار
# ══════════════════════════════════════════════════════════════

def _plot_grid_search_heatmap(unique_alphas, unique_l1_ratios, heat_matrix,
                              best_alpha, best_l1_ratio, save_dir):
    """
    نمودار چپ: Grid Search Heatmap (alpha × l1_ratio)
    
    پارامترها:
        unique_alphas: مقادیر یکتای alpha
        unique_l1_ratios: مقادیر یکتای l1_ratio
        heat_matrix: ماتریس ۲ بعدی CV MSE
        best_alpha: بهترین alpha
        best_l1_ratio: بهترین l1_ratio
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # رسم heatmap
    sns.heatmap(
        heat_matrix,
        cmap='YlOrRd',
        xticklabels=[f'{a:.0e}' for a in unique_alphas],
        yticklabels=[f'{l:.2f}' for l in unique_l1_ratios],
        ax=ax,
        cbar_kws={'label': 'CV MSE'}
    )
    
    # پیدا کردن ایندکس بهترین ترکیب
    best_alpha_idx = np.where(unique_alphas == best_alpha)[0][0]
    best_l1_idx = np.where(unique_l1_ratios == best_l1_ratio)[0][0]
    
    # رسم ستاره روی بهترین ترکیب
    ax.plot(
        best_alpha_idx + 0.5, best_l1_idx + 0.5,
        marker='*', markersize=20, color='gold',
        markeredgecolor='black', markeredgewidth=1.5,
        label=f'Best: α={best_alpha:.4f}, L1={best_l1_ratio:.2f}'
    )
    
    ax.set_xlabel('Alpha (log scale)', fontsize=12)
    ax.set_ylabel('L1 Ratio', fontsize=12)
    ax.set_title(
        'ElasticNet Grid Search — CV MSE Heatmap\n'
        '(Darker = Higher Error, Star = Best Combination)',
        fontweight='normal', fontsize=13
    )
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig12a_grid_search_heatmap.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


def _plot_en_coefficient_bar(coefficients, feature_names, zeroed_features, save_dir):
    """
    نمودار راست: ElasticNet Coefficient Bar Chart
    
    پارامترها:
        coefficients: آرایه ضرایب نهایی
        feature_names: لیست نام ویژگی‌ها
        zeroed_features: لیست ویژگی‌های حذف‌شده (ضریب صفر)
        save_dir: مسیر ذخیره
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # مرتب‌سازی بر اساس قدرمطلق (از بزرگ به کوچک)
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefs = coefficients[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # تعیین رنگ‌ها: آبی برای غیرصفر، خاکستری برای صفر
    colors = []
    for i, coef in enumerate(sorted_coefs):
        if sorted_names[i] in zeroed_features:
            colors.append('#cccccc')  # خاکستری برای حذف‌شده
        elif coef >= 0:
            colors.append('steelblue')  # آبی برای مثبت
        else:
            colors.append('salmon')  # قرمز برای منفی
    
    # رسم میله‌های افقی
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    
    # خط عمودی روی صفر
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Coefficient Value', fontsize=11)
    ax.set_title('ElasticNet Coefficients\n(Blue=Selected, Gray=Zeroed)', 
                 fontweight='normal', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # راهنمای رنگ
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Selected (non-zero)'),
        Patch(facecolor='#cccccc', label='Zeroed (removed)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig12b_coefficient_bar.png'
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
        color=C[5],
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
    save_path = save_dir / 'fig12c_actual_vs_predicted.png'
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
        boxprops=dict(facecolor=C[5], alpha=0.7),
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
        boxprops=dict(facecolor=C[6], alpha=0.7),
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
    save_path = save_dir / 'fig12d_cv_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"      ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۱۲
# ══════════════════════════════════════════════════════════════

def run_elasticnet_regression(
    phase4_result: Phase4Result,
    phase2_result: Phase2Result,
    phase5_result: Phase5Result,
    output_folder: str = 'phase12_elasticnet_regression_plots'
) -> Phase12Result:
    """
    اجرای کامل فاز ۱۲ — رگرسیون الاستیک نت
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_train, X_test)
        phase2_result: نتیجه فاز ۲ (شامل feature_names)
        phase5_result: نتیجه فاز ۵ (شامل cv و دیکشنری‌ها)
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase12Result: شامل بهترین (alpha, l1_ratio)، ضرایب و نتایج
    """
    MODEL_NAME = "ElasticNet"
    
    print("\n" + "=" * 65)
    print("📌  فاز ۱۲ — رگرسیون الاستیک نت (ElasticNet - L₁ + L₂)")
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
    print(f"   • Pipeline: StandardScaler → ElasticNet (max_iter=10000)")
    print(f"   • معیار GridSearch: neg_mean_squared_error")
    print(f"   • قدرت ElasticNet: ترکیب L₁ (انتخاب ویژگی) + L₂ (پایداری)")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: تعریف فضای جستجو دوبعدی و GridSearchCV
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🔍  بخش اول: جستجوی بهینه (alpha, l1_ratio)")
    print(f"{'─' * 65}")
    
    # ── تعریف بازه‌های جستجو ─────────────────────────────────
    alphas = np.logspace(-4, 2, 20)  # ۲۰ عدد لگاریتمی
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]  # ۷ مقدار دقیق
    
    n_combinations = len(alphas) * len(l1_ratios)
    
    print(f"\n▶  فضای جستجو دوبعدی:")
    print(f"   • تعداد مقادیر alpha: {len(alphas)}")
    print(f"   • محدوده alpha: {alphas[0]:.4f} تا {alphas[-1]:.4f}")
    print(f"   • مقادیر l1_ratio: {l1_ratios}")
    print(f"   • تعداد کل ترکیب‌ها: {n_combinations}")
    
    # ── ساخت Pipeline استاندارد ──────────────────────────────
    print(f"\n▶  ساخت Pipeline: StandardScaler → ElasticNet...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('en', ElasticNet(max_iter=10000))
    ])
    
    # ── تعریف گرید دوبعدی ────────────────────────────────────
    param_grid = {
        'en__alpha': alphas,
        'en__l1_ratio': l1_ratios
    }
    
    # ── اجرای GridSearchCV ───────────────────────────────────
    print(f"\n▶  اجرای GridSearchCV با {cv.n_splits}-Fold CV...")
    print(f"   • در حال آزمایش {n_combinations} ترکیب پارامتری...")
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # ── استخراج بهترین پارامترها ─────────────────────────────
    best_alpha = grid_search.best_params_['en__alpha']
    best_l1_ratio = grid_search.best_params_['en__l1_ratio']
    best_pipeline = grid_search.best_estimator_
    
    print(f"\n   ✅ بهترین alpha    : {best_alpha:.6f}")
    print(f"   ✅ بهترین l1_ratio: {best_l1_ratio:.2f}")
    print(f"   • بهترین CV MSE   : {-grid_search.best_score_:.4f}")
    
    # ── استخراج نتایج GridSearchCV ───────────────────────────
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['mean_test_mse'] = -cv_results_df['mean_test_score']
    cv_results_df['std_test_mse'] = cv_results_df['std_test_score']
    
    # ── ساخت ماتریس ۲ بعدی برای heatmap ──────────────────────
    unique_alphas = np.sort(alphas)
    unique_l1_ratios = np.sort(l1_ratios)
    
    heat_matrix = np.full((len(unique_l1_ratios), len(unique_alphas)), np.nan)
    
    for params, mean_mse in zip(cv_results_df['params'], cv_results_df['mean_test_mse']):
        alpha_idx = np.where(unique_alphas == params['en__alpha'])[0][0]
        l1_idx = np.where(unique_l1_ratios == params['en__l1_ratio'])[0][0]
        heat_matrix[l1_idx, alpha_idx] = mean_mse
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: ارزیابی مدل نهایی و شناسایی ویژگی‌های حذف‌شده
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🎯  بخش دوم: ارزیابی مدل نهایی")
    print(f"{'─' * 65}")
    
    # ── استخراج ضرایب نهایی ──────────────────────────────────
    coefficients = best_pipeline.named_steps['en'].coef_
    intercept = best_pipeline.named_steps['en'].intercept_
    
    # ── شناسایی ویژگی‌های انتخاب‌شده و حذف‌شده ──────────────
    selected_features = []
    zeroed_features = []
    
    for i, (feature_name, coef) in enumerate(zip(feature_names, coefficients)):
        if abs(coef) < 1e-10:
            zeroed_features.append(feature_name)
        else:
            selected_features.append(feature_name)
    
    n_selected = len(selected_features)
    n_zeroed = len(zeroed_features)
    
    # ── ساخت DataFrame ضرایب ─────────────────────────────────
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Status': ['Selected' if abs(c) >= 1e-10 else 'Zeroed' for c in coefficients]
    })
    coef_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_sorted = coef_sorted[['Feature', 'Coefficient', 'Status']].reset_index(drop=True)
    coef_sorted.index = coef_sorted.index + 1
    
    print(f"\n▶  جدول ضرایب (مرتب‌شده بر اساس قدرمطلق):")
    print("-" * 65)
    print(coef_sorted.to_string())
    print("-" * 65)
    
    # ── چاپ لیست ویژگی‌های انتخاب‌شده و حذف‌شده ──────────────
    print(f"\n▶  نتایج انتخاب ویژگی:")
    print(f"\n   ✅ ویژگی‌های انتخاب‌شده ({n_selected}):")
    if selected_features:
        for feat in selected_features:
            print(f"      • {feat}")
    else:
        print(f"      ⚠️ هیچ ویژگی‌ای انتخاب نشد!")
    
    print(f"\n   ❌ ویژگی‌های حذف‌شده ({n_zeroed}):")
    if zeroed_features:
        for feat in zeroed_features:
            print(f"      • {feat}")
    else:
        print(f"      ℹ️ هیچ ویژگی‌ای حذف نشد")
    
    print(f"\n   📊 نسبت انتخاب: {n_selected}/{len(feature_names)} ({n_selected/len(feature_names)*100:.1f}٪)")
    
    # ── تحلیل ضرایب ──────────────────────────────────────────
    print(f"\n▶  تحلیل ضرایب:")
    print(f"   • حداکثر ضریب: {coefficients.max():.4f}")
    print(f"   • حداقل ضریب: {coefficients.min():.4f}")
    print(f"   • میانگین قدرمطلق: {np.abs(coefficients).mean():.4f}")
    print(f"   • تعداد ضرایب مثبت: {(coefficients > 0).sum()}")
    print(f"   • تعداد ضرایب منفی: {(coefficients < 0).sum()}")
    print(f"   • تعداد ضرایب صفر: {n_zeroed}")
    
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
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: مقایسه با Ridge و Lasso
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش سوم: مقایسه با Ridge و Lasso")
    print(f"{'─' * 65}")
    
    comparison_with_ridge = {}
    comparison_with_lasso = {}
    
    # ── مقایسه با Ridge ──────────────────────────────────────
    if "Ridge" in phase5_result.models_results:
        ridge_result = phase5_result.models_results["Ridge"]
        ridge_coef = phase5_result.trained_models["Ridge"].named_steps['ridge'].coef_
        ridge_zeroed = (np.abs(ridge_coef) < 1e-10).sum()
        
        comparison_with_ridge = {
            'ridge_test_mse': ridge_result['test_mse'],
            'ridge_test_r2': ridge_result['test_r2'],
            'ridge_n_zeroed': ridge_zeroed,
            'en_test_mse': eval_result.test_mse,
            'en_test_r2': eval_result.test_r2,
            'en_n_zeroed': n_zeroed,
            'mse_diff': eval_result.test_mse - ridge_result['test_mse'],
            'r2_diff': eval_result.test_r2 - ridge_result['test_r2']
        }
        
        print(f"\n   📌 مقایسه با Ridge:")
        print(f"      • Ridge Test MSE  : {ridge_result['test_mse']:.4f}")
        print(f"      • EN Test MSE     : {eval_result.test_mse:.4f}")
        print(f"      • تفاوت MSE       : {comparison_with_ridge['mse_diff']:+.4f}")
        print(f"      • Ridge Test R²   : {ridge_result['test_r2']:.4f}")
        print(f"      • EN Test R²      : {eval_result.test_r2:.4f}")
        print(f"      • تفاوت R²        : {comparison_with_ridge['r2_diff']:+.4f}")
        print(f"      • Ridge ویژگی حذف : {ridge_zeroed}")
        print(f"      • EN ویژگی حذف    : {n_zeroed}")
        
        if comparison_with_ridge['r2_diff'] > 0:
            print(f"      ✅ ElasticNet بهتر از Ridge عمل کرد")
        else:
            print(f"      ⚠️ Ridge بهتر از ElasticNet عمل کرد")
    else:
        print(f"\n   ⚠️ نتایج Ridge در دسترس نیست")
    
    # ── مقایسه با Lasso ──────────────────────────────────────
    if "Lasso" in phase5_result.models_results:
        lasso_result = phase5_result.models_results["Lasso"]
        lasso_coef = phase5_result.trained_models["Lasso"].named_steps['lasso'].coef_
        lasso_zeroed = (np.abs(lasso_coef) < 1e-10).sum()
        
        comparison_with_lasso = {
            'lasso_test_mse': lasso_result['test_mse'],
            'lasso_test_r2': lasso_result['test_r2'],
            'lasso_n_zeroed': lasso_zeroed,
            'en_test_mse': eval_result.test_mse,
            'en_test_r2': eval_result.test_r2,
            'en_n_zeroed': n_zeroed,
            'mse_diff': eval_result.test_mse - lasso_result['test_mse'],
            'r2_diff': eval_result.test_r2 - lasso_result['test_r2']
        }
        
        print(f"\n   📌 مقایسه با Lasso:")
        print(f"      • Lasso Test MSE  : {lasso_result['test_mse']:.4f}")
        print(f"      • EN Test MSE     : {eval_result.test_mse:.4f}")
        print(f"      • تفاوت MSE       : {comparison_with_lasso['mse_diff']:+.4f}")
        print(f"      • Lasso Test R²   : {lasso_result['test_r2']:.4f}")
        print(f"      • EN Test R²      : {eval_result.test_r2:.4f}")
        print(f"      • تفاوت R²        : {comparison_with_lasso['r2_diff']:+.4f}")
        print(f"      • Lasso ویژگی حذف : {lasso_zeroed}")
        print(f"      • EN ویژگی حذف    : {n_zeroed}")
        
        if comparison_with_lasso['r2_diff'] > 0:
            print(f"      ✅ ElasticNet بهتر از Lasso عمل کرد")
        else:
            print(f"      ⚠️ Lasso بهتر از ElasticNet عمل کرد")
    else:
        print(f"\n   ⚠️ نتایج Lasso در دسترس نیست")
    
    # ── تحلیل ساختاری ────────────────────────────────────────
    print(f"\n   📌 تحلیل ساختاری ضرایب:")
    print(f"      • Ridge: ضرایب به صفر میل می‌کنند ولی صفر نمی‌شوند (L₂)")
    print(f"      • Lasso: ضرایب دقیقاً صفر می‌شوند (L₁) — انتخاب ویژگی قوی")
    print(f"      • ElasticNet: ترکیب هر دو — پایداری + انتخاب ویژگی")
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase12Result(
        best_alpha=best_alpha,
        best_l1_ratio=best_l1_ratio,
        best_pipeline=best_pipeline,
        coefficients=coefficients,
        feature_names=feature_names,
        coefficients_df=coef_sorted,
        selected_features=selected_features,
        zeroed_features=zeroed_features,
        n_selected=n_selected,
        n_zeroed=n_zeroed,
        grid_search_results={
            'best_alpha': best_alpha,
            'best_l1_ratio': best_l1_ratio,
            'best_score': -grid_search.best_score_,
            'n_alphas_tested': len(alphas),
            'n_l1_ratios_tested': len(l1_ratios),
            'n_total_combinations': n_combinations,
            'n_selected_features': n_selected,
            'n_zeroed_features': n_zeroed
        },
        cv_results_df=cv_results_df,
        heat_matrix=heat_matrix,
        unique_alphas=unique_alphas,
        unique_l1_ratios=unique_l1_ratios,
        evaluation_result=eval_result,
        comparison_with_ridge=comparison_with_ridge,
        comparison_with_lasso=comparison_with_lasso,
        model_name=MODEL_NAME
    )
    
    # ── ذخیره در دیکشنری‌های فاز ۵ ──────────────────────────
    phase5_result.models_results[MODEL_NAME] = evaluation_result_to_dict(eval_result)
    phase5_result.trained_models[MODEL_NAME] = best_pipeline
    
    # ذخیره اطلاعات اضافی
    phase5_result.models_results[MODEL_NAME]['best_alpha'] = best_alpha
    phase5_result.models_results[MODEL_NAME]['best_l1_ratio'] = best_l1_ratio
    phase5_result.models_results[MODEL_NAME]['selected_features'] = selected_features
    phase5_result.models_results[MODEL_NAME]['zeroed_features'] = zeroed_features
    
    print(f"\n   ✅ نتایج مدل '{MODEL_NAME}' در دیکشنری‌ها ذخیره شد.")
    
    # ── چاپ خلاصه معیارها ────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"📊  خلاصه معیارهای عملکردی:")
    print(f"{'─' * 65}")
    
    print(f"\n   📌 بهترین alpha    : {best_alpha:.6f}")
    print(f"   📌 بهترین l1_ratio: {best_l1_ratio:.2f}")
    print(f"   📌 ویژگی‌های انتخاب‌شده: {n_selected} از {len(feature_names)}")
    
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
    print(f"📈  رسم نمودارهای فاز ۱۲...")
    print(f"{'─' * 65}")
    
    # ── نمودار اول: پنل ۲ تایی ───────────────────────────────
    print(f"\n   ▶  رسم پنل ۲ تایی (Heatmap + Bar Chart)...")
    
    # چپ: Grid Search Heatmap
    _plot_grid_search_heatmap(
        unique_alphas, unique_l1_ratios, heat_matrix,
        best_alpha, best_l1_ratio, save_dir
    )
    
    # راست: Coefficient Bar Chart
    _plot_en_coefficient_bar(coefficients, feature_names, zeroed_features, save_dir)
    
    # ── نمودار دوم: AvP ──────────────────────────────────────
    print(f"\n   ▶  رسم Actual vs Predicted...")
    _plot_actual_vs_predicted(y_test, eval_result.y_pred_test, MODEL_NAME, save_dir)
    
    # ── نمودار سوم: Boxplot CV ───────────────────────────────
    print(f"\n   ▶  رسم Boxplot نتایج CV...")
    _plot_cv_boxplots(
        eval_result.cv_mse_scores,
        eval_result.cv_r2_scores,
        MODEL_NAME,
        save_dir
    )
    
    # ── پایان فاز ۱۲ ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"✅  فاز ۱۲ — رگرسیون الاستیک نت کامل شد.")
    print(f"   • بهترین alpha    : {best_alpha:.6f}")
    print(f"   • بهترین l1_ratio: {best_l1_ratio:.2f}")
    print(f"   • ویژگی‌های انتخاب‌شده: {n_selected}")
    print(f"   • ویژگی‌های حذف‌شده: {n_zeroed}")
    print("=" * 65)
    
    return result