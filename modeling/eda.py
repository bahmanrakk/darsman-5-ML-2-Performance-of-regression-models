"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۱ — بررسی اولیه داده (EDA)                             ║
║  Phase 1 — Exploratory Data Analysis                        ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
import math
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.config import TARGET_COLUMN, C


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | توضیحات فایل‌های داده
# ══════════════════════════════════════════════════════════════

DATASET_DESCRIPTIONS = {
    'used_8_cars_no_fe_after_corr_vif_final': {
        'name': 'No Feature Engineering',
        'short': 'no_fe',
        'description': (
            "این فایل شامل داده‌های خام خودروهای کارکرده است که:\n"
            "• هیچ مهندسی ویژگی (Feature Engineering) روی آن انجام نشده\n"
            "• ویژگی‌های با همبستگی بالا (Correlation) حذف شده‌اند\n"
            "• ویژگی‌های با VIF بالا (Multicollinearity) حذف شده‌اند\n"
            "• فقط ویژگی‌های اصلی و انتخاب‌شده استفاده شده‌اند"
        ),
        'color': '#E74C3C',
        'emoji': '🔴'
    },
    'used_7_cars_with_fe_after_corr_vif_final': {
        'name': 'With Feature Engineering',
        'short': 'with_fe',
        'description': (
            "این فایل شامل داده‌های با مهندسی ویژگی پیشرفته است که:\n"
            "• ویژگی‌های جدید و ترکیبی ساخته شده‌اند\n"
            "• نسبت‌های معنادار (مثل توان/حجم) جایگزین ویژگی‌های خام شده‌اند\n"
            "• ویژگی‌های با همبستگی بالا (Correlation) حذف شده‌اند\n"
            "• ویژگی‌های با VIF بالا (Multicollinearity) حذف شده‌اند\n"
            "• تبدیلات نرمال‌سازی (Log, Sqrt, BoxCox) اعمال شده‌اند"
        ),
        'color': '#2ECC71',
        'emoji': '🟢'
    }
}


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | تابع کمکی — فیلتر کردن ستون‌های باینری
# ══════════════════════════════════════════════════════════════

def _filter_binary_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    """
    حذف ستون‌های باینری (فقط ۰ و ۱) از لیست ستون‌های عددی
    
    پارامترها:
        df: دیتافریم ورودی
        numeric_cols: لیست ستون‌های عددی
    
    بازگشت:
        لیست ستون‌های غیرباینری
    """
    return [
        col for col in numeric_cols
        if not set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع ۱ — بررسی ساختار و کیفیت داده
# ══════════════════════════════════════════════════════════════

def analyze_data_structure(df: pd.DataFrame, dataset_info: dict) -> None:
    """
    بررسی ابعاد، نوع داده‌ها، مقادیر گمشده و آمار توصیفی
    """
    print("\n" + "=" * 65)
    print(f"📊  فاز ۱ — بررسی اولیه داده (EDA)")
    print("=" * 65)
    
    # چاپ اطلاعات فایل
    print(f"\n{dataset_info['emoji']}  فایل: {dataset_info['name']}")
    print(f"{'=' * 65}")
    print(dataset_info['description'])
    print("=" * 65)
    
    # ── ابعاد ────────────────────────────────────────────────
    print(f"\n▶  ابعاد  : {df.shape[0]:,} سطر × {df.shape[1]} ستون")
    
    # ── پنج سطر اول (Head) ───────────────────────────────────
    print(f"\n▶  پنج سطر اول داده:")
    print(df.head().to_string())
    
    # ── اطلاعات کامل (Info) ──────────────────────────────────
    print(f"\n▶  اطلاعات کامل ستون‌ها (df.info):")
    buffer = StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    
    # ── مقادیر گمشده ─────────────────────────────────────────
    missing = df.isnull().sum()
    print(f"\n▶  مقادیر گمشده:")
    if missing.sum() == 0:
        print("   ✅ هیچ مقدار گمشده‌ای یافت نشد.")
    else:
        print(missing[missing > 0].to_string())
    
    # ── آمار توصیفی ──────────────────────────────────────────
    print(f"\n▶  خلاصه آماری:")
    print(df.describe().round(3).to_string())
    
    print("\n" + "=" * 65)


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | تابع ۲ — رسم نمودار توزیع متغیر هدف
# ══════════════════════════════════════════════════════════════

def plot_target_distribution(df: pd.DataFrame, save_dir: Path, 
                            dataset_info: dict) -> None:
    """
    رسم هیستوگرام و باکس‌پلوت متغیر هدف
    """
    print(f"\n📈  رسم نمودار توزیع متغیر هدف ({dataset_info['short']})...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    target_series = df[TARGET_COLUMN]
    
    # ── هیستوگرام ────────────────────────────────────────────
    axes[0].hist(
        target_series, bins=40, color=dataset_info['color'], 
        edgecolor='white', alpha=0.85
    )
    axes[0].set_title(
        f'Target Histogram\n{TARGET_COLUMN}',
        fontweight='normal', fontsize=12
    )
    axes[0].set(xlabel=TARGET_COLUMN, ylabel='Count')
    
    # ── باکس‌پلوت ─────────────────────────────────────────────
    sns.boxplot(y=target_series, ax=axes[1], color=dataset_info['color'])
    axes[1].set_title(
        f'Target Boxplot\n{TARGET_COLUMN}',
        fontweight='normal', fontsize=12
    )
    axes[1].set(ylabel=TARGET_COLUMN)
    
    # عنوان اصلی
    fig.suptitle(
        f"'{dataset_info['name']}'\n"
        f"Phase 1 | Distribution of Target Variable",
        fontsize=14
    )
    plt.tight_layout()
    
    save_path = save_dir / f'fig01a_target_dist_{dataset_info["short"]}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۶ | تابع ۳ — رسم ماتریس همبستگی
# ══════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path,
                           dataset_info: dict) -> None:
    """
    رسم ماتریس همبستگی و چاپ همبستگی با متغیر هدف
    """
    print(f"\n🔥  رسم ماتریس همبستگی ({dataset_info['short']})...")
    
    # ── محاسبه ماتریس همبستگی ────────────────────────────────
    corr = df.corr(numeric_only=True)
    
    # ── چاپ همبستگی با متغیر هدف ─────────────────────────────
    print(f"\n▶  همبستگی هر ویژگی با '{TARGET_COLUMN}':")
    target_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    for feature, corr_val in target_corr.items():
        bar = '█' * int(abs(corr_val) * 20)
        print(f"   • {feature:35s} : {corr_val:+.4f} {bar}")
    
    # ── رسم نقشه حرارتی ──────────────────────────────────────
    n_c = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_c), max(6, n_c - 2)))
    
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        linewidths=0.4, annot_kws={'size': 8}, ax=ax
    )
    
    ax.set_title(
        f"'{dataset_info['name']}'\n"
        f"Phase 1 | Correlation Matrix",
        fontsize=14, pad=20
    )
    plt.tight_layout()
    
    save_path = save_dir / f'fig01b_heatmap_{dataset_info["short"]}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۷ | تابع ۴ — رسم باکس‌پلوت ویژگی‌ها
# ══════════════════════════════════════════════════════════════

def plot_features_boxplot(df: pd.DataFrame, save_dir: Path,
                         dataset_info: dict) -> None:
    """
    رسم باکس‌پلات برای ویژگی‌های عددی غیرباینری در قالب ردیف‌های پنج‌تایی
    """
    print(f"\n📦  رسم باکس‌پلات ویژگی‌ها ({dataset_info['short']})...")
    
    # انتخاب ستون‌های عددی (به جز متغیر هدف)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)
    
    # ❌ حذف ستون‌های باینری
    numeric_cols = _filter_binary_columns(df, numeric_cols)
    
    if len(numeric_cols) > 10:
        print(f"   ⚠️ تعداد ستون‌ها زیاد است ({len(numeric_cols)}). فقط ۱۰ ستون اول نمایش داده می‌شود.")
        numeric_cols = numeric_cols[:10]
    
    n_features = len(numeric_cols)
    if n_features == 0:
        print("   ⚠️ هیچ ویژگی عددی (غیر از هدف و غیر باینری) برای رسم وجود ندارد.")
        return
    
    cols_per_row = 5
    n_rows = math.ceil(n_features / cols_per_row)
    
    fig = plt.figure(figsize=(3 * cols_per_row, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, cols_per_row, figure=fig)
    
    axes = []
    for r in range(n_rows):
        start = r * cols_per_row
        end = min(start + cols_per_row, n_features)
        n_cols_in_row = end - start
        offset = (cols_per_row - n_cols_in_row) // 2
        
        for j in range(n_cols_in_row):
            col_idx = j + offset
            ax = fig.add_subplot(gs[r, col_idx])
            axes.append(ax)
    
    for i, ax in enumerate(axes):
        col = numeric_cols[i]
        sns.boxplot(y=df[col], ax=ax, color=dataset_info['color'])
        ax.set_title(col, fontweight='normal')
        ax.set_ylabel('')
    
    fig.suptitle(
        f"'{dataset_info['name']}'\n"
        f"Phase 1 | Features Boxplot",
        fontsize=14, fontweight='normal'
    )
    plt.tight_layout()
    
    save_path = save_dir / f'fig01c_features_boxplot_{dataset_info["short"]}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  بخش ۸ | تابع ۵ — رسم نمودار جفتی (Pairplot)
# ══════════════════════════════════════════════════════════════

def plot_pairplot(df: pd.DataFrame, save_dir: Path,
                 dataset_info: dict) -> None:
    """
    رسم نمودار جفتی برای ویژگی‌های مهم و متغیر هدف
    فقط در صورتی که تعداد ویژگی‌های غیرباینری کمتر از ۸ باشد
    """
    print(f"\n🔗  بررسی امکان رسم نمودار جفتی ({dataset_info['short']})...")
    
    # انتخاب ستون‌های عددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ✅ فیلتر ستون‌های باینری (مشابه باکس‌پلات)
    numeric_cols = _filter_binary_columns(df, numeric_cols)
    
    if len(numeric_cols) <= 8:
        print(f"   ✅ تعداد ستون‌های غیرباینری ({len(numeric_cols)}) مناسب است. رسم pairplot...")
        
        g = sns.pairplot(
            df[numeric_cols], 
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 20, 'color': dataset_info['color']},
            diag_kws={'color': dataset_info['color']}
        )
        g.fig.suptitle(
            f"{dataset_info['emoji']} {dataset_info['name']}\n"
            f"Phase 1 | Pairplot (non-binary features only)",
            fontsize=14, fontweight='bold', y=1.02
        )
        
        save_path = save_dir / f'fig01d_pairplot_{dataset_info["short"]}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"   ✅ ذخیره شد: {save_path.name}")
        
        plt.show()
    else:
        print(f"   ⚠️ تعداد ستون‌های غیرباینری ({len(numeric_cols)}) زیاد است. pairplot رسم نمی‌شود.")


# ══════════════════════════════════════════════════════════════
#  بخش ۹ | تابع اصلی — اجرای کامل فاز ۱
# ══════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, 
            dataset_name: str,
            output_folder: str = 'eda_plots') -> None:
    """
    اجرای کامل فاز ۱ — بررسی اولیه داده
    
    پارامترها:
        df: دیتافریم ورودی
        dataset_name: نام فایل داده (کلید در DATASET_DESCRIPTIONS)
        output_folder: نام پوشه ذخیره نمودارها
    """
    # دریافت اطلاعات فایل
    if dataset_name in DATASET_DESCRIPTIONS:
        dataset_info = DATASET_DESCRIPTIONS[dataset_name]
    else:
        dataset_info = {
            'name': dataset_name,
            'short': dataset_name[:10],
            'description': 'اطلاعات این فایل در دسترس نیست.',
            'color': '#3498DB',
            'emoji': '🔵'
        }
    
    # ایجاد پوشه خروجی
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print(f"📁  پوشه ذخیره نمودارها: {save_dir}")
    print("=" * 65)
    
    # اجرای مراحل
    analyze_data_structure(df, dataset_info)
    plot_target_distribution(df, save_dir, dataset_info)
    plot_correlation_matrix(df, save_dir, dataset_info)
    plot_features_boxplot(df, save_dir, dataset_info)
    plot_pairplot(df, save_dir, dataset_info)
    
    print("\n" + "=" * 65)
    print(f"✅  فاز ۱ — EDA برای {dataset_info['name']} کامل شد.")
    print("=" * 65)