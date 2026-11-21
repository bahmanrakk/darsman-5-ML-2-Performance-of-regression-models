"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۶ — تابع ارزیابی یکپارچه (Unified Evaluation Function) ║
║  Phase 6 — Unified Model Evaluation Function                ║
╚══════════════════════════════════════════════════════════════╝

هدف:
    ایجاد یک ساختار محاسباتی واحد و استاندارد که تمام مدل‌های پروژه
    از طریق آن ارزیابی شوند تا از هرگونه سوگیری یا ناهمگونی جلوگیری شود.

ترتیب اجرا (بسیار مهم):
    ۱) اجرای Cross-Validation روی مدل خام (قبل از fit)
    ۲) Fit نهایی روی کل Train
    ۳) پیش‌بینی روی Train و Test
    ۴) محاسبه زمان و معیارها

تغییرات فاز ۱۶:
    • اضافه شدن فیلد y_true_test به EvaluationResult
    • ذخیره خودکار y_test در نتیجه ارزیابی
    • پشتیبانی از رسم نمودارهای Actual vs Predicted و Residuals
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | Imports
# ══════════════════════════════════════════════════════════════

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

# اضافه کردن مسیر پروژه
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from modeling.cross_validator import Phase5Result


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | ساختار داده برای نتیجه ارزیابی
# ══════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """
    نتایج ارزیابی یک مدل
    
    نکته مهم dataclass:
        فیلدهای بدون مقدار پیش‌فرض باید قبل از فیلدهای با مقدار پیش‌فرض
        قرار بگیرند. به همین دلیل y_true_test در انتهای کلاس قرار گرفته است.
    
    تغییرات فاز ۱۶:
        • اضافه شدن y_true_test برای ذخیره مقادیر واقعی هدف تست
        • این فیلد برای رسم نمودارهای Actual vs Predicted و Residuals استفاده می‌شود
    """
    
    # ── فیلدهای بدون مقدار پیش‌فرض (اول) ─────────────────────
    
    # پیش‌بینی‌ها
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    
    # معیارهای Test
    test_mse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    
    # معیارهای Train
    train_mse: float
    train_r2: float
    
    # نتایج CV
    cv_mse_scores: np.ndarray
    cv_r2_scores: np.ndarray
    cv_mse_mean: float
    cv_mse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    
    # زمان
    train_time: float
    pred_time: float
    
    # ── فیلدهای با مقدار پیش‌فرض (آخر) ───────────────────────
    
    # اطلاعات اضافی (اختیاری)
    model_params: Optional[Dict[str, Any]] = None
    
    # ✅ جدید (فاز ۱۶): مقادیر واقعی هدف تست
    # برای رسم نمودارهای Actual vs Predicted و Residuals
    # در انتهای کلاس قرار گرفته تا ترتیب dataclass حفظ شود
    y_true_test: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | تابع ارزیابی یکپارچه
# ══════════════════════════════════════════════════════════════

def evaluate_model(model,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  phase5_result: Phase5Result,
                  model_name: str = "Model",
                  verbose: bool = True) -> EvaluationResult:
    """
    تابع ارزیابی یکپارچه برای تمام مدل‌های پروژه
    
    ترتیب اجرا (بسیار مهم برای جلوگیری از Data Leakage):
    ۱) اجرای Cross-Validation روی مدل خام (قبل از fit)
    ۲) Fit نهایی مدل روی کل داده‌های آموزشی
    ۳) پیش‌بینی روی Train و Test
    ۴) محاسبه زمان و معیارها
    
    پارامترها:
        model: مدل sklearn (fit نشده)
        X_train: ماتریس ویژگی‌های آموزش
        y_train: بردار هدف آموزش
        X_test: ماتریس ویژگی‌های آزمون
        y_test: بردار هدف آزمون
        phase5_result: نتیجه فاز ۵ (شامل شیء CV)
        model_name: نام مدل (برای گزارش)
        verbose: آیا گزارش چاپ شود؟
    
    بازگشت:
        EvaluationResult: شامل تمام معیارها، پیش‌بینی‌ها و مقادیر واقعی
    """
    cv = phase5_result.cv
    
    if verbose:
        print(f"\n{'─' * 65}")
        print(f"🔍  ارزیابی مدل: {model_name}")
        print(f"{'─' * 65}")
    
    # ══════════════════════════════════════════════════════════
    #  مرحله ۱: اجرای Cross-Validation (قبل از fit)
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n   ▶  مرحله ۱: اجرای {cv.n_splits}-Fold CV...")
    
    # CV برای MSE
    cv_neg_mse = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    cv_mse_scores = -cv_neg_mse
    
    # CV برای R²
    cv_r2_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    
    if verbose:
        print(f"      CV MSE: {cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}")
        print(f"      CV R² : {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    
    # ══════════════════════════════════════════════════════════
    #  مرحله ۲: Fit نهایی روی کل Train
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n   ▶  مرحله ۲: آموزش نهایی روی کل Train...")
    
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    
    if verbose:
        print(f"      زمان آموزش: {train_time:.5f} ثانیه")
    
    # ══════════════════════════════════════════════════════════
    #  مرحله ۳: پیش‌بینی
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n   ▶  مرحله ۳: پیش‌بینی روی Train و Test...")
    
    t0 = time.perf_counter()
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    pred_time = time.perf_counter() - t0
    
    if verbose:
        print(f"      زمان پیش‌بینی: {pred_time:.5f} ثانیه")
    
    # ══════════════════════════════════════════════════════════
    #  مرحله ۴: محاسبه معیارها
    # ══════════════════════════════════════════════════════════
    if verbose:
        print(f"\n   ▶  مرحله ۴: محاسبه معیارهای ارزیابی...")
    
    # معیارهای Test
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # معیارهای Train
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    if verbose:
        print(f"\n      📊 معیارهای Test:")
        print(f"         MSE  : {test_mse:.4f}")
        print(f"         RMSE : {test_rmse:.4f}")
        print(f"         MAE  : {test_mae:.4f}")
        print(f"         R²   : {test_r2:.4f}")
        
        print(f"\n      📊 معیارهای Train:")
        print(f"         MSE  : {train_mse:.4f}")
        print(f"         R²   : {train_r2:.4f}")
        
        print(f"\n      📊 نتایج CV:")
        print(f"         MSE Mean: {cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}")
        print(f"         R² Mean : {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    
    # ══════════════════════════════════════════════════════════
    #  ساخت نتیجه
    # ══════════════════════════════════════════════════════════
    result = EvaluationResult(
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        test_mse=test_mse,
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,
        train_mse=train_mse,
        train_r2=train_r2,
        cv_mse_scores=cv_mse_scores,
        cv_r2_scores=cv_r2_scores,
        cv_mse_mean=cv_mse_scores.mean(),
        cv_mse_std=cv_mse_scores.std(),
        cv_r2_mean=cv_r2_scores.mean(),
        cv_r2_std=cv_r2_scores.std(),
        train_time=round(train_time, 5),
        pred_time=round(pred_time, 5),
        y_true_test=y_test  # ✅ جدید (فاز ۱۶): ذخیره مقادیر واقعی
    )
    
    if verbose:
        print(f"\n   ✅ ارزیابی مدل '{model_name}' کامل شد.")
        print(f"   📌 y_true_test ذخیره شد (شکل: {y_test.shape})")
    
    return result


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تابع کمکی — تبدیل نتیجه به دیکشنری
# ══════════════════════════════════════════════════════════════

def evaluation_result_to_dict(result: EvaluationResult,
                             y_true_test: np.ndarray = None) -> Dict[str, Any]:
    """
    تبدیل EvaluationResult به دیکشنری برای ذخیره در RESULTS
    
    پارامترها:
        result: نتیجه ارزیابی
        y_true_test: مقادیر واقعی هدف تست (اختیاری — اگر None باشد،
                    از result.y_true_test استفاده می‌شود)
    
    بازگشت:
        دیکشنری شامل تمام معیارها و پیش‌بینی‌ها
    """
    # ✅ اولویت با پارامتر ورودی، سپس با فیلد داخل result
    y_true_final = y_true_test if y_true_test is not None else result.y_true_test
    
    result_dict = {
        # پیش‌بینی‌ها
        'y_pred_train': result.y_pred_train,
        'y_pred_test': result.y_pred_test,
        
        # ✅ مقادیر واقعی (برای فاز ۱۶ — رسم نمودارهای AvP و Residuals)
        'y_true_test': y_true_final,
        
        # Test
        'test_mse': result.test_mse,
        'test_rmse': result.test_rmse,
        'test_mae': result.test_mae,
        'test_r2': result.test_r2,
        
        # Train
        'train_mse': result.train_mse,
        'train_r2': result.train_r2,
        
        # CV
        'cv_mse_scores': result.cv_mse_scores,
        'cv_r2_scores': result.cv_r2_scores,
        'cv_mse_mean': result.cv_mse_mean,
        'cv_mse_std': result.cv_mse_std,
        'cv_r2_mean': result.cv_r2_mean,
        'cv_r2_std': result.cv_r2_std,
        
        # زمان
        'train_time': result.train_time,
        'pred_time': result.pred_time,
    }
    
    return result_dict


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | تابع اصلی — اجرای فاز ۶
# ══════════════════════════════════════════════════════════════

def run_phase_6() -> None:
    """
    اجرای فاز ۶ — معرفی تابع ارزیابی یکپارچه
    """
    print("\n" + "=" * 65)
    print("⚙️  فاز ۶ — تابع ارزیابی یکپارچه")
    print("=" * 65)
    
    print("\n▶  ترتیب اجرای تابع evaluate_model():")
    print("   ۱) Cross-Validation روی مدل خام (قبل از fit)")
    print("   ۲) Fit نهایی روی کل Train")
    print("   ۳) پیش‌بینی روی Train و Test")
    print("   ۴) محاسبه زمان و معیارها")
    
    print("\n▶  معیارهای خروجی تابع:")
    print("   • Test: MSE, RMSE, MAE, R²")
    print("   • Train: MSE, R²")
    print("   • CV: MSE Mean, MSE Std, R² Mean, R² Std")
    print("   • زمان: Training Time, Prediction Time")
    
    print("\n▶  ویژگی‌های فاز ۱۶:")
    print("   • ذخیره خودکار y_true_test در EvaluationResult")
    print("   • پشتیبانی از رسم نمودارهای Actual vs Predicted")
    print("   • پشتیبانی از رسم نمودارهای Residuals")
    
    print("\n" + "=" * 65)
    print("✅  فاز ۶ — تابع evaluate_model() آماده استفاده است.")
    print("=" * 65)


# اجرای خودکار هنگام import
if __name__ == "__main__":
    run_phase_6()