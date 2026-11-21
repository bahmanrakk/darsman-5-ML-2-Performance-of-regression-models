"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۵ — تعریف استراتژی ارزیابی و معیارهای پروژه            ║
║  Phase 5 — Evaluation Strategy & Metrics Definition         ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import KFold

from modeling.config import (
    RANDOM_STATE, CV_FOLDS
)
from modeling.data_splitter import Phase4Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | تعریف معیارهای ارزیابی پروژه
# ══════════════════════════════════════════════════════════════

METRICS = {
    'MSE': {
        'name': 'Mean Squared Error',
        'description': 'میانگین مربعات خطا - خطاهای بزرگ وزن بیشتری می‌گیرند',
        'direction': 'lower_is_better',
        'symbol': '↓ کمتر بهتر'
    },
    'RMSE': {
        'name': 'Root Mean Squared Error',
        'description': 'ریشه دوم MSE - بازگشت به واحد متغیر هدف برای تفسیر آسان',
        'direction': 'lower_is_better',
        'symbol': '↓ کمتر بهتر'
    },
    'MAE': {
        'name': 'Mean Absolute Error',
        'description': 'میانگین قدرمطلق خطا - حساسیت کمتر به داده‌های پرت',
        'direction': 'lower_is_better',
        'symbol': '↓ کمتر بهتر'
    },
    'R2': {
        'name': 'R-squared (Coefficient of Determination)',
        'description': 'ضریب تعیین - درصد واریانس تبیین‌شده توسط مدل (بین ۰ تا ۱)',
        'direction': 'higher_is_better',
        'symbol': '↑ بیشتر بهتر'
    }
}


def print_metrics_definition():
    """چاپ تعریف معیارهای ارزیابی پروژه"""
    print("\n" + "=" * 65)
    print("📊  فاز ۵ (بخش دوم) — تعریف معیارهای ارزیابی")
    print("=" * 65)
    
    print("\n▶  چهار معیار استاندارد پروژه:")
    for metric_key, metric_info in METRICS.items():
        print(f"\n   📌 {metric_key}: {metric_info['name']}")
        print(f"      {metric_info['description']}")
        print(f"      جهت بهینه‌سازی: {metric_info['symbol']}")
    
    print("\n" + "=" * 65)
    print("✅  معیارهای ارزیابی تعریف شدند.")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | ساختار داده برای نتیجه فاز ۵
# ══════════════════════════════════════════════════════════════

@dataclass
class Phase5Result:
    """نتایج فاز ۵ — تنظیمات ارزیابی و CV"""
    
    # فیلدهای بدون مقدار پیش‌فرض (اول)
    cv: KFold
    n_splits: int
    shuffle: bool
    random_state: int
    train_samples: int
    train_features: int
    
    # فیلدهای با مقدار پیش‌فرض (آخر)
    models_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trained_models: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع اصلی فاز ۵ — تعریف استراتژی CV + معیارها
# ══════════════════════════════════════════════════════════════

def setup_cross_validation(phase4_result: Phase4Result,
                          n_splits: int = CV_FOLDS,
                          shuffle: bool = True,
                          random_state: int = RANDOM_STATE) -> Phase5Result:
    """
    فاز ۵: تعریف استراتژی ارزیابی و اعتبارسنجی متقابل
    
    پارامترها:
        phase4_result: نتیجه فاز ۴ (شامل X_train, y_train)
        n_splits: تعداد فولدها (پیش‌فرض: از config.py)
        shuffle: آیا داده‌ها قبل از تقسیم به‌هم ریخته شوند؟
        random_state: seed برای تکرارپذیری
    
    بازگشت:
        Phase5Result: شامل شیء CV و دیکشنری‌های خالی
    """
    print("\n" + "=" * 65)
    print("⚙️  فاز ۵ — تعریف استراتژی ارزیابی و Cross-Validation")
    print("=" * 65)
    
    # ── بخش اول: تعریف شیء KFold ─────────────────────────────
    print(f"\n▶  تنظیمات Cross-Validation:")
    print(f"   • نوع         : KFold")
    print(f"   • تعداد فولدها: {n_splits}")
    print(f"   • Shuffle     : {shuffle}")
    print(f"   • Random State: {random_state}")
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    # ── تأیید مرز داده‌ها ────────────────────────────────────
    print(f"\n▶  تأیید مرز داده‌ها (قاعده حیاتی):")
    print(f"   • CV فقط روی داده‌های آموزشی (X_train) اعمال می‌شود")
    print(f"   • داده‌های آزمون (X_test) تا ارزیابی نهایی دست‌نخورده می‌مانند")
    print(f"   • هدف: جلوگیری از نشت داده (Data Leakage)")
    
    # ── نمایش اندازه فولدها ──────────────────────────────────
    X_train = phase4_result.X_train
    y_train = phase4_result.y_train
    
    print(f"\n▶  اندازه فولدها:")
    fold_size = len(X_train) // n_splits
    remainder = len(X_train) % n_splits
    
    print(f"   • کل نمونه‌های آموزشی: {len(X_train):,}")
    print(f"   • اندازه تقریبی هر فولد: {fold_size:,}")
    if remainder > 0:
        print(f"   • {remainder} فولد یک نمونه بیشتر خواهند داشت")
    
    print(f"\n   • تعداد ویژگی‌ها: {X_train.shape[1]}")
    
    # ── نمایش ساختار فولدها ──────────────────────────────────
    print(f"\n▶  ساختار فولدها:")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        print(f"   • فولد {fold_idx}: Train={len(train_idx):,} | Validation={len(val_idx):,}")
    
    # ── ایجاد دیکشنری‌های ذخیره‌سازی ─────────────────────────
    print(f"\n▶  آماده‌سازی مخازن ذخیره‌سازی:")
    models_results = {}
    trained_models = {}
    
    print(f"   ✅ دیکشنری 'models_results' ایجاد شد (برای نتایج ارزیابی)")
    print(f"   ✅ دیکشنری 'trained_models' ایجاد شد (برای مدل‌های آموزش‌دیده)")
    
    # ── بخش دوم: تعریف معیارهای ارزیابی ──────────────────────
    print_metrics_definition()
    
    # ── ساخت نتیجه ───────────────────────────────────────────
    result = Phase5Result(
        cv=cv,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        train_samples=len(X_train),
        train_features=X_train.shape[1],
        models_results=models_results,
        trained_models=trained_models
    )
    
    # ── خلاصه نهایی ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📋  خلاصه استراتژی ارزیابی:")
    print("=" * 65)
    print(f"   • روش ارزیابی     : {n_splits}-Fold Cross-Validation")
    print(f"   • داده‌های آموزشی  : {result.train_samples:,} نمونه")
    print(f"   • داده‌های آزمون   : {phase4_result.test_size:,} نمونه (دست‌نخورده)")
    print(f"   • تعداد ویژگی‌ها  : {result.train_features}")
    print(f"   • تکرارپذیری      : ✅ (Random State = {random_state})")
    print(f"   • جلوگیری از Leakage: ✅")
    print(f"   • معیارهای ارزیابی : MSE, RMSE, MAE, R²")
    print("=" * 65)
    
    print("\n✅  فاز ۵ — تعریف استراتژی CV و معیارها کامل شد.")
    
    return result


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | تابع کمکی — ذخیره نتایج مدل
# ══════════════════════════════════════════════════════════════

def store_model_results(phase5_result: Phase5Result,
                       model_name: str,
                       model: Any,
                       cv_scores: np.ndarray,
                       test_metrics: Dict[str, float]) -> None:
    """
    ذخیره نتایج یک مدل در دیکشنری‌های فاز ۵
    
    پارامترها:
        phase5_result: نتیجه فاز ۵
        model_name: نام مدل
        model: مدل آموزش‌دیده
        cv_scores: نمرات CV
        test_metrics: معیارهای تست
    """
    phase5_result.models_results[model_name] = {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_metrics': test_metrics
    }
    
    phase5_result.trained_models[model_name] = model
    
    print(f"   ✅ نتایج مدل '{model_name}' ذخیره شد")


# ══════════════════════════════════════════════════════════════
#  بخش ۶ | تابع کمکی — نمایش خلاصه نتایج
# ══════════════════════════════════════════════════════════════

def print_results_summary(phase5_result: Phase5Result) -> None:
    """
    چاپ خلاصه نتایج تمام مدل‌های ذخیره‌شده
    
    پارامترها:
        phase5_result: نتیجه فاز ۵
    """
    print("\n" + "=" * 65)
    print("📊  خلاصه نتایج تمام مدل‌ها")
    print("=" * 65)
    
    if not phase5_result.models_results:
        print("   ⚠️ هنوز هیچ مدلی ارزیابی نشده است.")
        return
    
    print(f"\n{'مدل':<25} {'CV Mean':<12} {'CV Std':<12} {'Test R²':<12} {'Test RMSE':<12}")
    print("-" * 65)
    
    for model_name, results in phase5_result.models_results.items():
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        test_r2 = results['test_metrics'].get('r2', 0)
        test_rmse = results['test_metrics'].get('rmse', 0)
        
        print(f"{model_name:<25} {cv_mean:<12.4f} {cv_std:<12.4f} {test_r2:<12.4f} {test_rmse:<12.2f}")
    
    print("=" * 65)