"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱۳ — مقایسه ساختاری ضرایب مدل‌های خطی                 ║
║  Phase 13 — Structural Coefficient Comparison               ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    تحلیل رفتاری و مقایسه عینی نحوه توزیع وزن‌ها و ضرایب در چهار
    مدل خطی مختلف برای درک استراتژی هر الگوریتم در مواجهه با متغیرها.

مدل‌های مقایسه‌شونده:
    ۱. Multivariate Linear Regression (بدون Regularization)
    ۲. Ridge Regression (L₂ Regularization)
    ۳. Lasso Regression (L₁ Regularization)
    ۴. ElasticNet Regression (L₁ + L₂ Regularization)

مدل‌های نمایش جداگانه:
    ۵. Simple Linear Regression (تک‌متغیره — فقط یک ضریب)
    ۶. Polynomial Regression (چندجمله‌ای — ویژگی‌های توانی)

خروجی‌ها:
    • جدول ۱: مقایسه ۴ مدل خطی (Multivariate, Ridge, Lasso, ElasticNet)
    • جدول ۲: اطلاعات Simple LR (ضریب شیب، عرض از مبدأ، معادله خط)
    • جدول ۳: خلاصه Polynomial (درجه بهینه، تعداد ویژگی‌ها، معیارها)
    • هشدار متدولوژیک درباره تفاوت مقیاس
    • لیست ویژگی‌های حذف‌شده توسط Lasso و ElasticNet
    • نمودار Grouped Bar Chart (۴ میله برای هر ویژگی)
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.config import C, MODEL_COLOR_MAP
from modeling.cross_validator import Phase5Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه فاز ۱۳
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase13Result:
    """نتایج فاز ۱۳ — مقایسه ضرایب مدل‌های خطی"""
    
    # جدول مقایسه ۴ مدل خطی
    coefficients_df: pd.DataFrame
    
    # ضرایب هر مدل (به صورت دیکشنری)
    model_coefficients: Dict[str, np.ndarray]
    
    # ویژگی‌های حذف‌شده
    lasso_zeroed_features: List[str]
    elasticnet_zeroed_features: List[str]
    
    # نام مدل‌های مقایسه‌شده
    compared_models: List[str]
    
    # اطلاعات Simple LR
    simple_lr_info: Optional[Dict] = None
    
    # اطلاعات Polynomial
    polynomial_info: Optional[Dict] = None


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | توابع کمکی
# ══════════════════════════════════════════════════════════════

def _extract_coefficients(phase5_result: Phase5Result) -> Dict[str, np.ndarray]:
    """
    استخراج ضرایب از مدل‌های آموزش‌دیده (فقط ۴ مدل خطی)
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل trained_models)
    
    بازگشت:
        دیکشنری شامل ضرایب هر مدل
    """
    coefficients = {}
    
    # ۱. Multivariate Linear Regression
    if "Multivariate LR" in phase5_result.trained_models:
        model = phase5_result.trained_models["Multivariate LR"]
        coefficients["Multivariate LR"] = model.coef_
    
    # ۲. Ridge Regression
    if "Ridge" in phase5_result.trained_models:
        model = phase5_result.trained_models["Ridge"]
        coefficients["Ridge"] = model.named_steps['ridge'].coef_
    
    # ۳. Lasso Regression
    if "Lasso" in phase5_result.trained_models:
        model = phase5_result.trained_models["Lasso"]
        coefficients["Lasso"] = model.named_steps['lasso'].coef_
    
    # ۴. ElasticNet Regression
    if "ElasticNet" in phase5_result.trained_models:
        model = phase5_result.trained_models["ElasticNet"]
        coefficients["ElasticNet"] = model.named_steps['en'].coef_
    
    return coefficients


def _find_zeroed_features(coefficients: np.ndarray, feature_names: list, 
                         threshold: float = 1e-10) -> List[str]:
    """
    شناسایی ویژگی‌های حذف‌شده (ضریب صفر)
    """
    zeroed = []
    for name, coef in zip(feature_names, coefficients):
        if abs(coef) < threshold:
            zeroed.append(name)
    return zeroed


def _extract_simple_lr_info(phase5_result: Phase5Result) -> Optional[Dict]:
    """
    استخراج اطلاعات Simple LR
    
    پارامترها:
        phase5_result: نتیجه فاز ۵
    
    بازگشت:
        دیکشنری شامل ضریب شیب، عرض از مبدأ، معادله خط
    """
    if "Simple LR" not in phase5_result.trained_models:
        return None
    
    model = phase5_result.trained_models["Simple LR"]
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # نام ویژگی از نتایج مدل
    feature_name = "Unknown"
    if "Simple LR" in phase5_result.models_results:
        # تلاش برای استخراج نام ویژگی از نام مدل یا نتایج
        pass
    
    equation = f"y = {slope:.4f} × X + {intercept:.4f}"
    
    return {
        'slope': slope,
        'intercept': intercept,
        'equation': equation,
        'feature_name': feature_name
    }


def _extract_polynomial_info(phase5_result: Phase5Result) -> Optional[Dict]:
    """
    استخراج اطلاعات Polynomial Regression
    
    پارامترها:
        phase5_result: نتیجه فاز ۵
    
    بازگشت:
        دیکشنری شامل درجه بهینه، تعداد ویژگی‌ها، معیارها
    """
    # پیدا کردن مدل Polynomial (نام آن "Polynomial (deg=X)" است)
    poly_model_name = None
    for name in phase5_result.trained_models.keys():
        if name.startswith("Polynomial"):
            poly_model_name = name
            break
    
    if poly_model_name is None:
        return None
    
    # استخراج درجه از نام مدل
    degree = None
    if "deg=" in poly_model_name:
        try:
            degree = int(poly_model_name.split("deg=")[1].split(")")[0])
        except:
            degree = None
    
    # استخراج معیارهای ارزیابی
    eval_result = phase5_result.models_results.get(poly_model_name, {})
    
    # استخراج تعداد ویژگی‌های تولیدشده
    n_features = None
    if poly_model_name in phase5_result.trained_models:
        pipeline = phase5_result.trained_models[poly_model_name]
        if hasattr(pipeline, 'named_steps') and 'poly' in pipeline.named_steps:
            n_features = pipeline.named_steps['poly'].n_output_features_
    
    return {
        'model_name': poly_model_name,
        'degree': degree,
        'n_features_generated': n_features,
        'test_mse': eval_result.get('test_mse', None),
        'test_r2': eval_result.get('test_r2', None),
        'test_rmse': eval_result.get('test_rmse', None),
        'cv_mse_mean': eval_result.get('cv_mse_mean', None),
        'cv_r2_mean': eval_result.get('cv_r2_mean', None)
    }


def _plot_grouped_bar_chart(coefficients_df: pd.DataFrame, save_dir: Path) -> None:
    """
    رسم نمودار Grouped Bar Chart برای مقایسه ۴ مدل خطی
    """
    # استخراج داده‌ها
    feature_names = coefficients_df['Feature'].tolist()
    n_features = len(feature_names)
    n_models = len(coefficients_df.columns) - 1  # منهای ستون Feature
    
    # رنگ‌های اختصاصی برای هر مدل
    model_colors = {
        'Linear (Multivariate)': 'steelblue',
        'Ridge': '#E67E22',
        'Lasso': '#9B59B6',
        'ElasticNet': '#16A085'
    }
    
    # ایجاد figure
    fig, ax = plt.subplots(figsize=(max(14, n_features * 0.8), 7))
    
    # موقعیت میله‌ها
    x = np.arange(n_features)
    width = 0.2  # عرض هر میله
    
    # رسم میله‌ها برای هر مدل
    for i, col in enumerate(coefficients_df.columns[1:]):  # منهای Feature
        color = model_colors.get(col, C[i])
        values = coefficients_df[col].values
        
        # بررسی ضرایب صفر برای رنگ خاکستری
        colors = []
        for val in values:
            if abs(val) < 1e-10:
                colors.append('#cccccc')  # خاکستری برای صفر
            else:
                colors.append(color)
        
        ax.bar(
            x + i * width - (n_models - 1) * width / 2,
            values,
            width,
            label=col,
            color=colors,
            edgecolor='white',
            linewidth=0.5
        )
    
    # تنظیمات محور
    ax.set_xlabel('Features', fontsize=12, fontweight='normal')
    ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='normal')
    ax.set_title(
        'Phase 13 | Coefficient Comparison — Multivariate vs Ridge vs Lasso vs ElasticNet\n'
        '(Gray bars = Zero coefficients, i.e., removed features)',
        fontsize=13, fontweight='normal', pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    
    # خط افقی روی صفر
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    
    # راهنما
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    save_path = save_dir / 'fig13_coefficient_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۱۳
# ══════════════════════════════════════════════════════════════

def run_coefficient_comparison(
    phase5_result: Phase5Result,
    feature_names: list,
    output_folder: str = 'phase13_coefficient_comparison_plots'
) -> Phase13Result:
    """
    اجرای کامل فاز ۱۳ — مقایسه ساختاری ضرایب
    
    پارامترها:
        phase5_result: نتیجه فاز ۵ (شامل trained_models)
        feature_names: لیست نام ویژگی‌ها
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase13Result: شامل جدول ضرایب و نتایج مقایسه
    """
    print("\n" + "=" * 65)
    print("📌  فاز ۱۳ — مقایسه ساختاری ضرایب مدل‌های خطی")
    print("=" * 65)
    
    # ── ایجاد پوشه خروجی ────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁  پوشه ذخیره نمودارها: {save_dir}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش اول: جمع‌آوری ضرایب ۴ مدل خطی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش اول: جمع‌آوری ضرایب از ۴ مدل خطی")
    print(f"{'─' * 65}")
    
    # استخراج ضرایب
    model_coefficients = _extract_coefficients(phase5_result)
    
    # بررسی وجود مدل‌ها
    compared_models = list(model_coefficients.keys())
    n_models = len(compared_models)
    
    print(f"\n▶  مدل‌های خطی یافت‌شده: {n_models}")
    for model_name in compared_models:
        print(f"   • {model_name}")
    
    if n_models < 4:
        print(f"\n   ⚠️ هشدار: همه ۴ مدل خطی در دسترس نیستند.")
        print(f"      مدل‌های موجود: {compared_models}")
    
    # ══════════════════════════════════════════════════════════
    #  بخش دوم: ساخت جدول مقایسه ۴ مدل خطی
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📋  بخش دوم: جدول مقایسه ۴ مدل خطی")
    print(f"{'─' * 65}")
    
    # ساخت DataFrame
    data = {'Feature': feature_names}
    
    for model_name in compared_models:
        col_name = model_name
        if model_name == "Multivariate LR":
            col_name = "Linear (Multivariate)"
        
        data[col_name] = model_coefficients[model_name]
    
    coefficients_df = pd.DataFrame(data)
    
    # گرد کردن اعداد
    for col in coefficients_df.columns[1:]:
        coefficients_df[col] = coefficients_df[col].round(4)
    
    print(f"\n▶  جدول مقایسه ۴ مدل خطی:")
    print("-" * 100)
    print(coefficients_df.to_string(index=False))
    print("-" * 100)
    
    # ══════════════════════════════════════════════════════════
    #  بخش سوم: استخراج اطلاعات Simple LR
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📌  بخش سوم: اطلاعات Simple Linear Regression")
    print(f"{'─' * 65}")
    
    simple_lr_info = _extract_simple_lr_info(phase5_result)
    
    if simple_lr_info is not None:
        print(f"\n▶  Simple LR (تک‌متغیره):")
        print(f"   • ضریب شیب (Slope)       : {simple_lr_info['slope']:.4f}")
        print(f"   • عرض از مبدأ (Intercept): {simple_lr_info['intercept']:.4f}")
        print(f"   • معادله خط              : {simple_lr_info['equation']}")
        print(f"   • ویژگی استفاده‌شده      : {simple_lr_info['feature_name']}")
        
        # جدول خلاصه Simple LR
        simple_lr_df = pd.DataFrame({
            'Parameter': ['Slope', 'Intercept', 'Equation'],
            'Value': [
                f"{simple_lr_info['slope']:.4f}",
                f"{simple_lr_info['intercept']:.4f}",
                simple_lr_info['equation']
            ]
        })
        print(f"\n▶  جدول خلاصه Simple LR:")
        print(simple_lr_df.to_string(index=False))
    else:
        print(f"\n   ⚠️ مدل Simple LR در دسترس نیست")
    
    # ══════════════════════════════════════════════════════════
    #  بخش چهارم: استخراج اطلاعات Polynomial
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📌  بخش چهارم: اطلاعات Polynomial Regression")
    print(f"{'─' * 65}")
    
    polynomial_info = _extract_polynomial_info(phase5_result)
    
    if polynomial_info is not None:
        print(f"\n▶  Polynomial Regression (چندجمله‌ای):")
        print(f"   • نام مدل                : {polynomial_info['model_name']}")
        print(f"   • درجه بهینه             : {polynomial_info['degree']}")
        print(f"   • تعداد ویژگی‌های تولیدشده: {polynomial_info['n_features_generated']}")
        
        if polynomial_info['test_mse'] is not None:
            print(f"\n   📊 معیارهای ارزیابی:")
            print(f"      • Test MSE  : {polynomial_info['test_mse']:.4f}")
            print(f"      • Test RMSE : {polynomial_info['test_rmse']:.4f}")
            print(f"      • Test R²   : {polynomial_info['test_r2']:.4f}")
            print(f"      • CV MSE    : {polynomial_info['cv_mse_mean']:.4f}")
            print(f"      • CV R²     : {polynomial_info['cv_r2_mean']:.4f}")
        
        # جدول خلاصه Polynomial
        poly_data = {
            'Parameter': ['Model Name', 'Degree', 'Generated Features', 
                         'Test MSE', 'Test R²', 'CV MSE', 'CV R²'],
            'Value': [
                polynomial_info['model_name'],
                str(polynomial_info['degree']),
                str(polynomial_info['n_features_generated']),
                f"{polynomial_info['test_mse']:.4f}" if polynomial_info['test_mse'] else "N/A",
                f"{polynomial_info['test_r2']:.4f}" if polynomial_info['test_r2'] else "N/A",
                f"{polynomial_info['cv_mse_mean']:.4f}" if polynomial_info['cv_mse_mean'] else "N/A",
                f"{polynomial_info['cv_r2_mean']:.4f}" if polynomial_info['cv_r2_mean'] else "N/A"
            ]
        }
        poly_df = pd.DataFrame(poly_data)
        print(f"\n▶  جدول خلاصه Polynomial:")
        print(poly_df.to_string(index=False))
        
        print(f"\n   ℹ️  توجه: ضرایب Polynomial مربوط به ویژگی‌های توانی هستند")
        print(f"      و با ضرایب مدل‌های خطی قابل مقایسه مستقیم نیستند.")
    else:
        print(f"\n   ⚠️ مدل Polynomial در دسترس نیست")
    
    # ══════════════════════════════════════════════════════════
    #  بخش پنجم: هشدار متدولوژیک
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"⚠️  بخش پنجم: هشدار متدولوژیک")
    print(f"{'─' * 65}")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  توجه مهم: تفاوت مقیاس ضرایب                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ضرایب مدل رگرسیون خطی چندمتغیره (Multivariate LR) بر روی داده‌های ║
║  خام و بدون مقیاس‌بندی محاسبه شده‌اند.                               ║
║                                                                      ║
║  در مقابل، ضرایب مدل‌های Ridge، Lasso و ElasticNet بر روی داده‌های  ║
║  استانداردشده (Standardized) به دست آمده‌اند.                        ║
║                                                                      ║
║  لذا مقایسه مستقیم مقادیر عددی خام آن‌ها بدون نرمال‌سازی اصولی،     ║
║  گمراه‌کننده خواهد بود.                                              ║
║                                                                      ║
║  برای مقایسه عادلانه، باید:                                          ║
║  • یا همه مدل‌ها را روی داده استانداردشده آموزش داد                  ║
║  • یا ضرایب را با فرمول زیر به مقیاس اصلی برگرداند:                  ║
║                                                                      ║
║      coef_original = coef_standardized / std(feature)                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # ══════════════════════════════════════════════════════════
    #  بخش ششم: شناسایی ویژگی‌های حذف‌شده
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"🔍  بخش ششم: شناسایی ویژگی‌های حذف‌شده")
    print(f"{'─' * 65}")
    
    # Lasso
    lasso_zeroed = []
    if "Lasso" in model_coefficients:
        lasso_zeroed = _find_zeroed_features(
            model_coefficients["Lasso"], feature_names
        )
        print(f"\n   📌 ویژگی‌های حذف‌شده توسط Lasso ({len(lasso_zeroed)}):")
        if lasso_zeroed:
            for feat in lasso_zeroed:
                print(f"      • {feat}")
        else:
            print(f"      ℹ️ هیچ ویژگی‌ای حذف نشد")
    
    # ElasticNet
    elasticnet_zeroed = []
    if "ElasticNet" in model_coefficients:
        elasticnet_zeroed = _find_zeroed_features(
            model_coefficients["ElasticNet"], feature_names
        )
        print(f"\n   📌 ویژگی‌های حذف‌شده توسط ElasticNet ({len(elasticnet_zeroed)}):")
        if elasticnet_zeroed:
            for feat in elasticnet_zeroed:
                print(f"      • {feat}")
        else:
            print(f"      ℹ️ هیچ ویژگی‌ای حذف نشد")
    
    # Ridge (برای تأکید بر اینکه ضرایب صفر نمی‌شوند)
    ridge_zeroed = []
    if "Ridge" in model_coefficients:
        ridge_zeroed = _find_zeroed_features(
            model_coefficients["Ridge"], feature_names
        )
        print(f"\n   📌 ویژگی‌های حذف‌شده توسط Ridge ({len(ridge_zeroed)}):")
        if ridge_zeroed:
            for feat in ridge_zeroed:
                print(f"      • {feat}")
        else:
            print(f"      ✅ Ridge هیچ ویژگی‌ای را حذف نمی‌کند (L₂ فقط ضرایب را کوچک می‌کند)")
    
    # ══════════════════════════════════════════════════════════
    #  بخش هفتم: تحلیل ساختاری
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📊  بخش هفتم: تحلیل ساختاری ضرایب")
    print(f"{'─' * 65}")
    
    print(f"\n   📌 استراتژی هر مدل در مواجهه با متغیرها:")
    print(f"\n   ۱. Simple LR (تک‌متغیره):")
    print(f"      • فقط یک ضریب برای بهترین ویژگی")
    print(f"      • ساده‌ترین مدل (Baseline)")
    print(f"      • نادیده گرفتن سایر ویژگی‌ها")
    
    print(f"\n   ۲. Multivariate LR (بدون Regularization):")
    print(f"      • تمام ضرایب غیرصفر هستند")
    print(f"      • حساس به هم‌خطی (Multicollinearity)")
    print(f"      • ممکن است ضرایب بسیار بزرگ یا کوچک داشته باشد")
    
    print(f"\n   ۳. Ridge (L₂ Regularization):")
    print(f"      • ضرایب به سمت صفر میل می‌کنند ولی صفر نمی‌شوند")
    print(f"      • پایداری در شرایط هم‌خطی شدید")
    print(f"      • هیچ ویژگی‌ای حذف نمی‌شود")
    
    print(f"\n   ۴. Lasso (L₁ Regularization):")
    print(f"      • ضرایب دقیقاً به صفر می‌رسند")
    print(f"      • انتخاب ویژگی خودکار (Feature Selection)")
    print(f"      • تعداد ویژگی‌های حذف‌شده: {len(lasso_zeroed)}")
    
    print(f"\n   ۵. ElasticNet (L₁ + L₂):")
    print(f"      • ترکیب پایداری Ridge و انتخاب ویژگی Lasso")
    print(f"      • ضرایب به صفر می‌رسند (مثل Lasso)")
    print(f"      • تعداد ویژگی‌های حذف‌شده: {len(elasticnet_zeroed)}")
    
    print(f"\n   ۶. Polynomial (چندجمله‌ای):")
    if polynomial_info and polynomial_info['degree']:
        print(f"      • درجه بهینه: {polynomial_info['degree']}")
        print(f"      • تولید ویژگی‌های توانی (x², x³, x₁x₂, ...)")
        print(f"      • تعداد ویژگی‌های تولیدشده: {polynomial_info['n_features_generated']}")
        print(f"      • قابلیت مدل‌سازی روابط غیرخطی")
        print(f"      • ضرایب مربوط به ویژگی‌های توانی (نه ویژگی‌های اصلی)")
    
    # ══════════════════════════════════════════════════════════
    #  ساخت نتیجه
    # ══════════════════════════════════════════════════════════
    result = Phase13Result(
        coefficients_df=coefficients_df,
        model_coefficients=model_coefficients,
        lasso_zeroed_features=lasso_zeroed,
        elasticnet_zeroed_features=elasticnet_zeroed,
        compared_models=compared_models,
        simple_lr_info=simple_lr_info,
        polynomial_info=polynomial_info
    )
    
    # ══════════════════════════════════════════════════════════
    #  رسم نمودار
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"📈  رسم نمودار Grouped Bar Chart (۴ مدل خطی)...")
    print(f"{'─' * 65}")
    
    _plot_grouped_bar_chart(coefficients_df, save_dir)
    
    # ── پایان فاز ۱۳ ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  فاز ۱۳ — مقایسه ساختاری ضرایب کامل شد.")
    print(f"   • ۴ مدل خطی مقایسه شدند")
    if simple_lr_info:
        print(f"   • Simple LR: Slope = {simple_lr_info['slope']:.4f}")
    if polynomial_info:
        print(f"   • Polynomial: Degree = {polynomial_info['degree']}, Features = {polynomial_info['n_features_generated']}")
    print("=" * 65)
    
    return result