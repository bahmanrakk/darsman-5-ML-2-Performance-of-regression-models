"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۲ و ۳ — تعریف X و y و انتخاب بهترین ویژگی             ║
║  Phase 2 & 3 — Feature-Target Split & Best Feature Select  ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.config import TARGET_COLUMN, C
from modeling.eda import DATASET_DESCRIPTIONS  # ✅ import در بالای فایل


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتایج فاز ۲ و ۳
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase2Result:
    """نتایج فاز ۲ — تعریف X و y"""
    X_df: pd.DataFrame                 # ✅ ماتریس ویژگی‌ها به صورت DataFrame
    y: np.ndarray                      # بردار هدف
    feature_names: list                # لیست نام ویژگی‌ها
    target_name: str                   # نام متغیر هدف
    n_features: int                    # تعداد ویژگی‌ها
    n_samples: int                     # تعداد نمونه‌ها


@dataclass
class Phase3Result:
    """نتایج فاز ۳ — انتخاب بهترین ویژگی"""
    best_feature_name: str             # نام ویژگی برنده
    best_feature_idx: int              # ایندکس ویژگی برنده
    best_feature_corr: float           # همبستگی مطلق با هدف
    correlations: pd.Series            # همبستگی تمام ویژگی‌ها


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | فاز ۲ — تعریف X و y
# ══════════════════════════════════════════════════════════════

def split_features_target(df: pd.DataFrame, 
                         target_column: str = TARGET_COLUMN) -> Phase2Result:
    """
    فاز ۲: جداسازی متغیر هدف از ویژگی‌های مستقل
    
    پارامترها:
        df: دیتافریم ورودی
        target_column: نام ستون هدف
    
    بازگشت:
        Phase2Result: شامل X (DataFrame), y, feature_names و اطلاعات دیگر
    """
    print("\n" + "=" * 65)
    print("📐  فاز ۲ — تعریف X و y (تفکیک ویژگی‌ها و هدف)")
    print("=" * 65)
    
    # ── بررسی وجود ستون هدف ──────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(f"❌ ستون هدف '{target_column}' در دیتافریم یافت نشد!")
    
    # ── جداسازی متغیر هدف ────────────────────────────────────
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])  # ✅ به صورت DataFrame
    feature_names = X_df.columns.tolist()
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase2Result(
        X_df=X_df,
        y=y,
        feature_names=feature_names,
        target_name=target_column,
        n_features=len(feature_names),
        n_samples=len(y)
    )
    
    # ── چاپ گزارش ────────────────────────────────────────────
    print(f"\n▶  متغیر هدف  : {result.target_name}")
    print(f"   • شکل y     : {result.y.shape}  (تأیید یک‌بعدی بودن: {'✅' if result.y.ndim == 1 else '❌'})")
    print(f"   • تعداد نمونه‌ها: {result.n_samples:,}")
    
    print(f"\n▶  ماتریس ویژگی‌ها (X):")
    print(f"   • شکل X_df  : {result.X_df.shape}")
    print(f"   • تعداد ویژگی‌ها: {result.n_features}")
    print(f"   • نوع X_df  : {type(result.X_df).__name__} (نام ستون‌ها حفظ شده‌اند)")
    
    print(f"\n▶  لیست نام ویژگی‌ها ({result.n_features} ستون):")
    for i, name in enumerate(result.feature_names, 1):
        print(f"      {i:2d}. {name}")
    
    # ── تأیید نهایی ──────────────────────────────────────────
    if target_column not in result.feature_names:
        print(f"\n✅  تأیید شد: ستون هدف '{target_column}' از X حذف شده است.")
    else:
        print(f"\n❌  هشدار: ستون هدف هنوز در X وجود دارد!")
    
    print("\n" + "=" * 65)
    print("✅  فاز ۲ — X و y با موفقیت تعریف شدند.")
    print("=" * 65)
    
    return result


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | فاز ۳ — انتخاب بهترین ویژگی
# ══════════════════════════════════════════════════════════════

def select_best_feature(corr_matrix: pd.DataFrame,
                       phase2_result: Phase2Result) -> Phase3Result:
    """
    فاز ۳: انتخاب ویژگی با بیشترین همبستگی مطلق با هدف
    
    پارامترها:
        corr_matrix: ماتریس همبستگی از فاز ۱ (برای جلوگیری از محاسبه تکراری)
        phase2_result: نتیجه فاز ۲
    
    بازگشت:
        Phase3Result: شامل اطلاعات ویژگی برنده
    """
    print("\n" + "=" * 65)
    print("🔍  فاز ۳ — انتخاب ویژگی برای رگرسیون تک‌متغیره")
    print("=" * 65)
    
    target_column = phase2_result.target_name
    
    # ── استفاده از ماتریس همبستگی موجود ──────────────────────
    correlations = corr_matrix[target_column].drop(target_column).abs()
    correlations_sorted = correlations.sort_values(ascending=False)
    
    # ── انتخاب ویژگی برنده ────────────────────────────────────
    best_feature_name = correlations_sorted.index[0]
    best_feature_corr = correlations_sorted.iloc[0]
    best_feature_idx = phase2_result.feature_names.index(best_feature_name)
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase3Result(
        best_feature_name=best_feature_name,
        best_feature_idx=best_feature_idx,
        best_feature_corr=best_feature_corr,
        correlations=correlations_sorted
    )
    
    # ── چاپ جدول رتبه‌بندی ────────────────────────────────────
    print(f"\n▶  رتبه‌بندی ویژگی‌ها بر اساس |همبستگی| با '{target_column}':")
    print("-" * 65)
    print(f"{'رتبه':<6} {'ویژگی':<35} {'|r|':<10} {'وضعیت'}")
    print("-" * 65)
    
    for rank, (feature, corr_val) in enumerate(correlations_sorted.items(), 1):
        if feature == best_feature_name:
            status = "🏆 برنده"
        elif corr_val > 0.7:
            status = "⭐ قوی"
        elif corr_val > 0.4:
            status = "● متوسط"
        else:
            status = "○ ضعیف"
        print(f"{rank:<6} {feature:<35} {corr_val:<10.4f} {status}")
    
    print("-" * 65)
    
    # ── چاپ نتیجه نهایی ──────────────────────────────────────
    print(f"\n🏆  ویژگی برنده:")
    print(f"   • نام     : '{result.best_feature_name}'")
    print(f"   • ایندکس  : {result.best_feature_idx}")
    print(f"   • |r|     : {result.best_feature_corr:.4f}")
    
    print("\n" + "=" * 65)
    print("✅  فاز ۳ — انتخاب ویژگی کامل شد.")
    print("=" * 65)
    
    return result


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | نمودار — میله‌ای همبستگی
# ══════════════════════════════════════════════════════════════

def plot_correlation_bar_chart(phase3_result: Phase3Result,
                               dataset_info: dict,
                               save_dir: Path) -> None:
    """
    رسم نمودار میله‌ای همبستگی تمام ویژگی‌ها با هدف
    """
    print(f"\n📊  رسم نمودار میله‌ای همبستگی ({dataset_info['short']})...")
    
    correlations = phase3_result.correlations
    
    # ── آماده‌سازی رنگ‌ها ────────────────────────────────────
    colors = []
    for feature in correlations.index:
        if feature == phase3_result.best_feature_name:
            colors.append('#FFD700')  # طلایی برای برنده
        elif correlations[feature] > 0.7:
            colors.append('#2ECC71')  # سبز برای قوی
        elif correlations[feature] > 0.4:
            colors.append('#3498DB')  # آبی برای متوسط
        else:
            colors.append('#95A5A6')  # خاکستری برای ضعیف
    
    # ── رسم نمودار ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(
        range(len(correlations)), 
        correlations.values,
        color=colors,
        edgecolor='white',
        linewidth=1.5
    )
    
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('Absolute Correlation with Target (|r|)', fontsize=11)
    ax.set_ylabel('Features', fontsize=11)
    ax.set_title(
        f"'{dataset_info['name']}'\n"
        f"Phase 3 | Feature Correlation with Target",
        fontsize=14, fontweight='normal', pad=15
    )
    
    # ── اضافه کردن مقادیر روی میله‌ها ─────────────────────────
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # ── خط آستانه ────────────────────────────────────────────
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.4, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    # ── راهنمای رنگ ──────────────────────────────────────────
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FFD700', label='🏆 Best Feature'),
        plt.Rectangle((0,0),1,1, facecolor='#2ECC71', label='⭐ Strong (|r|>0.7)'),
        plt.Rectangle((0,0),1,1, facecolor='#3498DB', label='● Moderate (0.4<|r|<0.7)'),
        plt.Rectangle((0,0),1,1, facecolor='#95A5A6', label='○ Weak (|r|<0.4)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    save_path = save_dir / f'fig03_correlation_bar_{dataset_info["short"]}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۶ | تابع اصلی — اجرای فاز ۲ و ۳
# ══════════════════════════════════════════════════════════════

def run_phase_2_3(df: pd.DataFrame,
                  corr_matrix: pd.DataFrame,
                  dataset_name: str,
                  output_folder: str = 'feature_selection_plots') -> tuple:
    """
    اجرای کامل فاز ۲ و ۳
    
    پارامترها:
        df: دیتافریم ورودی
        corr_matrix: ماتریس همبستگی از فاز ۱ (برای جلوگیری از محاسبه تکراری)
        dataset_name: نام فایل داده
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        tuple: (Phase2Result, Phase3Result)
    """
    # ── دریافت اطلاعات فایل ──────────────────────────────────
    if dataset_name in DATASET_DESCRIPTIONS:
        dataset_info = DATASET_DESCRIPTIONS[dataset_name]
    else:
        dataset_info = {
            'name': dataset_name,
            'short': dataset_name[:10],
            'color': '#3498DB',
            'emoji': '🔵'
        }
    
    # ── ایجاد پوشه خروجی ─────────────────────────────────────
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print(f"📁  پوشه ذخیره نمودارها: {save_dir}")
    print("=" * 65)
    
    # ── فاز ۲: تعریف X و y ───────────────────────────────────
    phase2_result = split_features_target(df, TARGET_COLUMN)
    
    # ── فاز ۳: انتخاب بهترین ویژگی (با استفاده از corr_matrix) ─
    phase3_result = select_best_feature(corr_matrix, phase2_result)
    
    # ── رسم نمودار میله‌ای ────────────────────────────────────
    plot_correlation_bar_chart(phase3_result, dataset_info, save_dir)
    
    print("\n" + "=" * 65)
    print(f"✅  فاز ۲ و ۳ برای {dataset_info['name']} کامل شدند.")
    print("=" * 65)
    
    return phase2_result, phase3_result