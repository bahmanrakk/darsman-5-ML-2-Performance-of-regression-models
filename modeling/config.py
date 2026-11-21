"""
╔══════════════════════════════════════════════════════════════╗
║  فاز ۰ — آماده‌سازی محیط و تنظیمات پایه پروژه              ║
║  Phase 0 — Environment Setup & Project Constants            ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  بخش ۱ | نادیده گرفتن هشدارها
# ══════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  بخش ۲ | کتابخانه‌های پایه (برای استفاده در سایر ماژول‌ها)
# ══════════════════════════════════════════════════════════════

# ── کار با داده ──────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── رسم نمودار ───────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── ابزارهای کمکی ────────────────────────────────────────────
import time
import random


# ══════════════════════════════════════════════════════════════
#  بخش ۳ | تنظیمات ثابت پروژه (Constants)
# ══════════════════════════════════════════════════════════════

RANDOM_STATE  = 42          # برای تکرارپذیری نتایج
TEST_SIZE     = 0.20        # ۲۰٪ داده برای آزمون
CV_FOLDS      = 5           # K در K-Fold CV
POLY_MAX_DEG  = 3           # حداکثر درجه آزمون در رگرسیون چندجمله‌ای

TARGET_COLUMN = 'selling_price'   # نام ستون هدف

# ── تنظیم seed جهانی برای تکرارپذیری کامل ────────────────────
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════
#  بخش ۴ | تنظیمات فونت فارسی
# ══════════════════════════════════════════════════════════════

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    def persian_text(text: str) -> str:
        return text
else:
    reshaping_config = {
        'delete_harakat': False,
        'delete_tatweel': False,
        'support_ligatures': True,
        'language': 'Farsi'
    }
    reshaper = arabic_reshaper.ArabicReshaper(configuration=reshaping_config)
    
    def persian_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        reshaped_text = reshaper.reshape(text)
        return get_display(reshaped_text)


# ══════════════════════════════════════════════════════════════
#  بخش ۵ | تنظیمات استایل نمودارها
# ══════════════════════════════════════════════════════════════

# ── انتخاب استایل پایه ───────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

# ── تنظیمات فونت ─────────────────────────────────────────────
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# ── تنظیمات رنگ و ظاهر ───────────────────────────────────────
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'
plt.rcParams['text.color'] = '#333F4B'
plt.rcParams['axes.labelcolor'] = '#333F4B'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ── تنظیمات اندازه و کیفیت ───────────────────────────────────
plt.rcParams.update({
    'figure.figsize'   : (13, 5),
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'figure.dpi'       : 100,
})

# ── پالت رنگ ثابت برای سراسر پروژه ──────────────────────────
C = sns.color_palette('tab10', 10)   # ۱۰ رنگ ثابت

# ── نقشه رنگ اختصاصی برای هر مدل ─────────────────────────────
MODEL_COLOR_MAP = {
    'Simple LR'      : C[0],  
    'Multivariate LR': C[1],
    'Polynomial'     : C[2],
    'Ridge'          : C[3],
    'Lasso'          : C[4],
    'ElasticNet'     : C[5],
}


# ══════════════════════════════════════════════════════════════
#  بخش ۶ | چاپ تأییدیه و مستندسازی تنظیمات
# ══════════════════════════════════════════════════════════════

def print_setup_info():
    """چاپ اطلاعات تنظیمات پروژه برای مستندسازی"""
    print("=" * 65)
    print("✅  فاز ۰ — کتابخانه‌ها و تنظیمات آماده شدند.")
    print("=" * 65)
    print("\n📋 تنظیمات ثابت پروژه:")
    print(f"   • RANDOM_STATE  : {RANDOM_STATE}")
    print(f"   • TEST_SIZE     : {TEST_SIZE}")
    print(f"   • CV_FOLDS      : {CV_FOLDS}")
    print(f"   • POLY_MAX_DEG  : {POLY_MAX_DEG}")
    print(f"   • TARGET_COLUMN : {TARGET_COLUMN}")
    print(f"   • استایل نمودار : seaborn-v0_8-whitegrid")
    print(f"   • فونت          : Tahoma")
    print(f"   • Seeds تنظیم‌شده: numpy ✅ | random ✅")
    print("=" * 65)


# اجرای خودکار هنگام import
print_setup_info()