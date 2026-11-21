"""
╔══════════════════════════════════════════════════════════════╗
           فاز ۴ — تقسیم داده‌ها (Train / Test Split)
                   Phase 4 — Data Splitting
╚══════════════════════════════════════════════════════════════╝
"""

# ============================================================
#  بخش ۱ | Imports
# ============================================================

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split

from modeling.config import (
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, C
)
from modeling.feature_selector import Phase2Result, Phase3Result


# ============================================================
#  بخش ۲ | ساختار داده برای نتیجه فاز ۴
# ============================================================

@dataclass
class Phase4Result:
    """نتایج فاز ۴ — تقسیم داده‌ها"""
    # ماتریس‌های کامل
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    
    # ماتریس‌های تک‌ویژگی (برای رگرسیون ساده)
    X_uni_train: np.ndarray
    X_uni_test: np.ndarray
    
    # آمار
    total_size: int
    train_size: int
    test_size: int
    train_percentage: float
    test_percentage: float


# ============================================================
#  بخش ۳ | تابع اصلی فاز ۴ — تقسیم داده‌ها
# ============================================================

def split_train_test(phase2_result: Phase2Result,
                    phase3_result: Phase3Result,
                    test_size: float = TEST_SIZE,
                    random_state: int = RANDOM_STATE) -> Phase4Result:
    """فاز ۴: تقسیم داده‌ها به مجموعه‌های آموزش و آزمون"""
    print("\n" + "=" * 65)
    print("✂️  فاز ۴ — تقسیم داده‌ها (Train / Test Split)")
    print("=" * 65)
    
    # ✅ استخراج داده‌ها - X حالا DataFrame است
    X_df = phase2_result.X_df
    y = phase2_result.y
    
    # ── تقسیم داده‌ها ────────────────────────────────────────
    print(f"\n▶  تنظیمات تقسیم:")
    print(f"   • نسبت آزمون  : {test_size*100:.0f}٪")
    print(f"   • نسبت آموزش  : {(1-test_size)*100:.0f}٪")
    print(f"   • RANDOM_STATE: {random_state}")
    
    # ✅ استفاده از train_test_split با DataFrame
    from sklearn.model_selection import train_test_split
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # ✅ تبدیل به numpy array برای سازگاری با sklearn
    X_train = X_train_df.values
    X_test = X_test_df.values
    
    # ── استخراج نسخه تک‌ویژگی ────────────────────────────────
    best_feat_idx = phase3_result.best_feature_idx
    best_feat_name = phase3_result.best_feature_name
    
    X_uni_train = X_train[:, best_feat_idx].reshape(-1, 1)
    X_uni_test = X_test[:, best_feat_idx].reshape(-1, 1)
    
    print(f"\n▶  استخراج نسخه تک‌ویژگی:")
    print(f"   • ویژگی برنده: '{best_feat_name}'")
    print(f"   • ایندکس     : {best_feat_idx}")
    print(f"   • شکل X_uni_train: {X_uni_train.shape}")
    print(f"   • شکل X_uni_test : {X_uni_test.shape}")
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    total_size = len(y)
    train_size = len(y_train)
    test_size_actual = len(y_test)
    
    result = Phase4Result(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_uni_train=X_uni_train,
        X_uni_test=X_uni_test,
        total_size=total_size,
        train_size=train_size,
        test_size=test_size_actual,
        train_percentage=(train_size / total_size) * 100,
        test_percentage=(test_size_actual / total_size) * 100
    )
    
    # ── چاپ گزارش ────────────────────────────────────────────
    print(f"\n▶  ابعاد ماتریس‌ها:")
    print(f"   • X_train : {result.X_train.shape}")
    print(f"   • X_test  : {result.X_test.shape}")
    print(f"   • y_train : {result.y_train.shape}")
    print(f"   • y_test  : {result.y_test.shape}")
    
    print(f"\n▶  آمار تقسیم:")
    print(f"   • کل داده      : {result.total_size:,} نمونه")
    print(f"   • آموزش (Train): {result.train_size:,} نمونه ({result.train_percentage:.1f}٪)")
    print(f"   • آزمون (Test) : {result.test_size:,} نمونه ({result.test_percentage:.1f}٪)")
    
    if abs(result.train_percentage - 80) < 1 and abs(result.test_percentage - 20) < 1:
        print(f"\n✅  تأیید شد: تقسیم ۸۰/۲۰ به درستی انجام شد.")
    
    print("\n" + "=" * 65)
    print("✅  فاز ۴ — تقسیم داده‌ها کامل شد.")
    print("=" * 65)
    
    return result

# ============================================================
#  بخش ۴ | نمودار — مقایسه توزیع هدف در Train و Test
# ============================================================

def plot_train_test_distribution(phase2_result: Phase2Result,
                                phase4_result: Phase4Result,
                                dataset_info: dict,
                                save_dir: Path) -> None:
    """
    رسم نمودار KDE برای مقایسه توزیع y_train و y_test
    
    پارامترها:
        phase2_result: نتیجه فاز ۲
        phase4_result: نتیجه فاز ۴
        dataset_info: اطلاعات فایل داده
        save_dir: مسیر ذخیره نمودار
    """
    print(f"\n📊  رسم نمودار مقایسه توزیع Train vs Test ({dataset_info['short']})...")
    
    y_train = phase4_result.y_train
    y_test = phase4_result.y_test
    target_name = phase2_result.target_name
    
    # -- رسم نمودار -------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # منحنی KDE برای Train
    sns.kdeplot(
        y_train,
        ax=ax,
        color=C[0],
        fill=True,
        alpha=0.4,
        linewidth=2,
        label=f'Train (n={len(y_train):,})'
    )
    
    # منحنی KDE برای Test
    sns.kdeplot(
        y_test,
        ax=ax,
        color=C[2],
        fill=True,
        alpha=0.4,
        linewidth=2,
        label=f'Test (n={len(y_test):,})'
    )
    
    # -- برچسب‌ها و عنوان --------------------------------------
    ax.set_xlabel(target_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        f"'{dataset_info['name']}'\n"
        f"Phase 4 | Target Distribution — Train vs Test",
        fontsize=14, fontweight='normal', pad=15
    )
    
    # -- آمار توصیفی در گوشه نمودار ---------------------------
    stats_text = (
        f"Train: μ={y_train.mean():.2f}, σ={y_train.std():.2f}\n"
        f"Test : μ={y_test.mean():.2f}, σ={y_test.std():.2f}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # -- ذخیره نمودار -----------------------------------------
    save_path = save_dir / f'fig04_split_dist_{dataset_info["short"]}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"   ✅ ذخیره شد: {save_path.name}")
    
    # -- تحلیل تطابق توزیع‌ها ---------------------------------
    mean_diff = abs(y_train.mean() - y_test.mean())
    std_diff = abs(y_train.std() - y_test.std())
    
    print(f"\n📈  تحلیل تطابق توزیع‌ها:")
    print(f"   • تفاوت میانگین (Train vs Test): {mean_diff:.4f}")
    print(f"   • تفاوت انحراف معیار           : {std_diff:.4f}")
    
    if mean_diff < y_train.std() * 0.1 and std_diff < y_train.std() * 0.1:
        print(f"   ✅ توزیع‌ها بسیار شبیه هستند — تقسیم‌بندی نماینده است.")
    else:
        print(f"   ⚠️ تفاوت قابل توجه در توزیع‌ها — تقسیم‌بندی را بررسی کنید.")
    
    plt.show()


# ============================================================
#  بخش ۵ | تابع اصلی — اجرای فاز ۴
# ============================================================

def run_phase_4(phase2_result: Phase2Result,
                phase3_result: Phase3Result,
                dataset_name: str,
                output_folder: str = 'data_split_plots') -> Phase4Result:
    """
    اجرای کامل فاز ۴
    
    پارامترها:
        phase2_result: نتیجه فاز ۲
        phase3_result: نتیجه فاز ۳
        dataset_name: نام فایل داده
        output_folder: نام پوشه ذخیره نمودارها
    
    بازگشت:
        Phase4Result: نتایج تقسیم داده‌ها
    """
    # -- دریافت اطلاعات فایل ----------------------------------
    from modeling.eda import DATASET_DESCRIPTIONS
    
    if dataset_name in DATASET_DESCRIPTIONS:
        dataset_info = DATASET_DESCRIPTIONS[dataset_name]
    else:
        dataset_info = {
            'name': dataset_name,
            'short': dataset_name[:10],
            'color': '#3498DB',
            'emoji': '🔵'
        }
    
    # -- ایجاد پوشه خروجی -------------------------------------
    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'data' / output_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print(f"📁  پوشه ذخیره نمودارها: {save_dir}")
    print("=" * 65)
    
    # -- اجرای فاز ۴ ------------------------------------------
    phase4_result = split_train_test(phase2_result, phase3_result)
    
    # -- رسم نمودار -------------------------------------------
    plot_train_test_distribution(phase2_result, phase4_result, dataset_info, save_dir)
    
    print("\n" + "=" * 65)
    print(f"✅  فاز ۴ برای {dataset_info['name']} کامل شد.")
    print("=" * 65)
    
    return phase4_result