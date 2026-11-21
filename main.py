"""
╔══════════════════════════════════════════════════════════════╗
║                   فایل اصلی اجرای پروژه                      ║
║                       Main Entry Point                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  تنظیم مسیر پروژه
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.file_io import read_csv_simple, save_csv_file
from pipeline import build_preprocessing_pipeline


# ══════════════════════════════════════════════════════════════
#  ثابت‌های پروژه (برای یکپارچگی در کل کدبیس)
# ══════════════════════════════════════════════════════════════

FILE_WITH_FE = "used_7_cars_with_fe_after_corr_vif_final.csv"
FILE_NO_FE   = "used_8_cars_no_fe_after_corr_vif_final.csv"

DATASET_KEY_WITH_FE = 'used_7_cars_with_fe_after_corr_vif_final'
DATASET_KEY_NO_FE   = 'used_8_cars_no_fe_after_corr_vif_final'


# ══════════════════════════════════════════════════════════════
#  تابع ۱ | اجرای پایپ‌لاین پیش‌پردازش
# ══════════════════════════════════════════════════════════════

def run_preprocessing_pipeline():
    """
    اجرای کامل پایپ‌لاین پیش‌پردازش برای تولید دو فایل نهایی.
    
    بازگشت:
        tuple: (df_with_fe, df_no_fe)
    """
    print("\n" + "=" * 65)
    print("🚀 شروع اجرای پایپ‌لاین پیش‌پردازش")
    print("=" * 65)

    # ── بارگذاری داده خام ────────────────────────────────────
    print("\n📂 مرحله 01_01: بارگذاری داده از فایل CSV")
    print("-" * 65)
    data = read_csv_simple("used_cars.csv")

    if data is None:
        print("❌ خطا در خواندن فایل. اجرای برنامه متوقف شد.")
        return None, None

    cols_to_remove = ['name']

    # ══════════════════════════════════════════════════════════
    #  مسیر ۱: داده‌های با مهندسی ویژگی (With FE)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("🟢 مسیر ۱: داده‌های با مهندسی ویژگی + نرمال‌سازی")
    print("=" * 65)

    final_columns_with_fe = [
        'annual_km', 'age_squared',
        'mileage_per_cc', 'power_per_cc', 'power_ratio', 'auto_high_power',
        'fuel_CNG', 'fuel_Diesel', 'fuel_LPG',
        'premium_seller', 'owner_ord',
        'selling_price'
    ]

    vif_columns_with_fe = [col for col in final_columns_with_fe if col != 'selling_price']

    pipeline_with_fe = build_preprocessing_pipeline(
        drop_torque=True,
        mileage_decimal_places=4,
        max_power_decimal_places=4,
        engine_decimal_places=4,
        columns_to_drop=cols_to_remove,
        run_outlier_analysis=False,
        final_columns=final_columns_with_fe,
        vif_analysis_columns=vif_columns_with_fe,
        run_vif_analysis=True,
        sqrt_columns=['annual_km', 'mileage_per_cc'],
        log_columns=['age_squared', 'power_ratio', 'auto_high_power', 'premium_seller'],
        boxcox_columns=['power_per_cc'],
        standardize_directly=['mileage'],
        run_normalization=True
    )

    print("\n⚙️ اجرای پایپ‌لاین (با FE + نرمال‌سازی)...")
    df_with_fe = pipeline_with_fe.fit_transform(data)
    save_csv_file(df_with_fe, FILE_WITH_FE)

    # ══════════════════════════════════════════════════════════
    #  مسیر ۲: داده‌های بدون مهندسی ویژگی (No FE)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("🔴 مسیر ۲: داده‌های بدون مهندسی ویژگی + نرمال‌سازی")
    print("=" * 65)

    final_columns_no_fe = [
        'km_driven', 'mileage', 'seats', 'age',
        'engine', 'max_power',
        'fuel_LPG', 'fuel_Petrol',
        'seller_type_Dealer', 'seller_type_Trustmark_Dealer',
        'transmission_bin', 'owner_ord',
        'selling_price'
    ]

    vif_columns_no_fe = [col for col in final_columns_no_fe if col != 'selling_price']

    pipeline_no_fe = build_preprocessing_pipeline(
        drop_torque=True,
        mileage_decimal_places=4,
        max_power_decimal_places=4,
        engine_decimal_places=4,
        columns_to_drop=cols_to_remove,
        run_outlier_analysis=False,
        final_columns=final_columns_no_fe,
        vif_analysis_columns=vif_columns_no_fe,
        run_vif_analysis=True,
        sqrt_columns=['km_driven'],
        log_columns=['age', 'engine', 'max_power'],
        boxcox_columns=[],
        standardize_directly=['mileage'],
        run_normalization=True
    )

    print("\n⚙️ اجرای پایپ‌لاین (بدون FE + نرمال‌سازی)...")
    df_no_fe = pipeline_no_fe.fit_transform(data)
    save_csv_file(df_no_fe, FILE_NO_FE)

    print("\n" + "=" * 65)
    print("✅ پایان اجرای پایپ‌لاین پیش‌پردازش")
    print(f"📁 فایل‌های ذخیره شده:\n   1. {FILE_WITH_FE}\n   2. {FILE_NO_FE}")
    print("=" * 65)

    return df_with_fe, df_no_fe


# ══════════════════════════════════════════════════════════════
#  تابع ۲ | اجرای فازهای مدل‌سازی (۰ تا ۱۷)
# ══════════════════════════════════════════════════════════════
def run_modeling_phases(df_with_fe=None, df_no_fe=None):
    """
    اجرای کامل فازهای مدل‌سازی از ۰ تا ۱۷ برای هر دو مجموعه داده.
    
    بازگشت:
        dict: شامل تمام نتایج و آبجکت‌های فازها
    """
    print("\n" + "=" * 65)
    print("🚀 شروع فازهای مدل‌سازی")
    print("=" * 65)
    
    # ── مقداردهی اولیه متغیرها (برای جلوگیری از خطای UnboundLocalError) ──
    phase_vars = [
        'phase2', 'phase3', 'phase4', 'phase5', 'phase7', 'phase8', 'phase9',
        'phase10', 'phase11', 'phase12', 'phase13', 'phase14', 'phase15',
        'phase16', 'phase17'
    ]
    for var in phase_vars:
        globals()[f"{var}_with_fe"] = globals()[f"{var}_no_fe"] = None

    # ── ایمپورت ماژول‌های مدل‌سازی ──────────────────────────
    from modeling.config import print_setup_info
    from modeling.data_preprocessor import preprocess_for_modeling
    from modeling.eda import run_eda
    from modeling.feature_selector import run_phase_2_3
    from modeling.data_splitter import run_phase_4
    from modeling.cross_validator import setup_cross_validation
    from modeling.evaluator import run_phase_6
    from modeling.simple_regression import run_simple_regression
    from modeling.multivariate_regression import run_multivariate_regression
    from modeling.polynomial_regression import run_polynomial_regression
    from modeling.ridge_regression import run_ridge_regression
    from modeling.lasso_regression import run_lasso_regression
    from modeling.elasticnet_regression import run_elasticnet_regression
    from modeling.coefficient_comparison import run_coefficient_comparison
    from modeling.overfitting_analysis import run_overfitting_analysis
    from modeling.master_table import run_master_table
    from modeling.final_visualizations import run_final_visualizations
    from modeling.executive_summary import run_executive_summary

    # ── تابع کمکی پردازش یک مجموعه داده ────────────────────
    def _process_dataset(df, dataset_key, file_path, suffix, label):
        """
        پردازش یک مجموعه داده از فاز ۱ تا ۱۷.
        
        پارامترها:
            df: دیتافریم ورودی (یا None برای بارگذاری از فایل)
            dataset_key: کلید DATASET_DESCRIPTIONS
            file_path: مسیر فایل CSV
            suffix: پسوند نام پوشه‌ها ('with_fe' یا 'no_fe')
            label: برچسب فارسی برای چاپ
        """
        # بارگذاری در صورت نیاز
        if df is None:
            print(f"\n📊 بارگذاری: {file_path}")
            df = read_csv_simple(file_path)
        
        if df is None:
            print(f"❌ فایل داده‌های {label} بارگذاری نشد.")
            return (None,) * 17
        
        # پیش‌پردازش اولیه
        df = preprocess_for_modeling(df)
        
        # فاز ۱: EDA
        print(f"\n{'='*65}\n🟢 فاز ۱: EDA برای داده‌های {label}\n{'='*65}")
        run_eda(
            df,
            dataset_name=dataset_key,
            output_folder=f'eda_plots_{suffix}'
        )
        
        # محاسبه ماتریس همبستگی (برای فاز ۲ و ۳)
        corr = df.corr(numeric_only=True)
        
        # فاز ۲ و ۳: انتخاب ویژگی
        print(f"\n{'='*65}\n🟢 فاز ۲ و ۳: انتخاب ویژگی\n{'='*65}")
        p2, p3 = run_phase_2_3(
            df,
            corr_matrix=corr,
            dataset_name=dataset_key,
            output_folder=f'feature_selection_plots_{suffix}'
        )
        
        # فاز ۴: تقسیم داده‌ها
        print(f"\n{'='*65}\n🟢 فاز ۴: تقسیم داده‌ها\n{'='*65}")
        p4 = run_phase_4(
            p2, p3,
            dataset_name=dataset_key,
            output_folder=f'data_split_plots_{suffix}'
        )
        
        # فاز ۵: استراتژی CV
        print(f"\n{'='*65}\n🟢 فاز ۵: تعریف استراتژی CV\n{'='*65}")
        p5 = setup_cross_validation(p4)
        
        # فاز ۶: تابع ارزیابی (فقط اطلاع‌رسانی)
        print(f"\n{'='*65}\n🟢 فاز ۶: تابع ارزیابی یکپارچه آماده است\n{'='*65}")
        run_phase_6()
        
        # فاز ۷: رگرسیون خطی تک‌متغیره
        print(f"\n{'='*65}\n🟢 فاز ۷: رگرسیون خطی تک‌متغیره (Simple LR)\n{'='*65}")
        p7 = run_simple_regression(
            p4, p3, p5,
            output_folder=f'phase7_simple_regression_plots_{suffix}'
        )
        
        # فاز ۸: رگرسیون خطی چندمتغیره
        print(f"\n{'='*65}\n🟢 فاز ۸: رگرسیون خطی چندمتغیره\n{'='*65}")
        p8 = run_multivariate_regression(
            p4, p2, p5,
            output_folder=f'phase8_multivariate_regression_plots_{suffix}'
        )
        
        # فاز ۹: رگرسیون چندجمله‌ای
        print(f"\n{'='*65}\n🟢 فاز ۹: رگرسیون چندجمله‌ای\n{'='*65}")
        p9 = run_polynomial_regression(
            p4, p5,
            output_folder=f'phase9_polynomial_regression_plots_{suffix}'
        )
        
        # فاز ۱۰: Ridge Regression
        print(f"\n{'='*65}\n🟢 فاز ۱۰: رگرسیون ریج\n{'='*65}")
        p10 = run_ridge_regression(
            p4, p2, p5,
            output_folder=f'phase10_ridge_regression_plots_{suffix}'
        )
        
        # فاز ۱۱: Lasso Regression
        print(f"\n{'='*65}\n🟢 فاز ۱۱: رگرسیون لاسو\n{'='*65}")
        p11 = run_lasso_regression(
            p4, p2, p5,
            output_folder=f'phase11_lasso_regression_plots_{suffix}'
        )
        
        # فاز ۱۲: ElasticNet Regression
        print(f"\n{'='*65}\n🟢 فاز ۱۲: رگرسیون الاستیک نت\n{'='*65}")
        p12 = run_elasticnet_regression(
            p4, p2, p5,
            output_folder=f'phase12_elasticnet_regression_plots_{suffix}'
        )
        
        # فاز ۱۳: مقایسه ضرایب
        print(f"\n{'='*65}\n🟢 فاز ۱۳: مقایسه ضرایب مدل‌های خطی\n{'='*65}")
        p13 = run_coefficient_comparison(
            p5,
            feature_names=p2.feature_names,
            output_folder=f'phase13_coefficient_comparison_plots_{suffix}'
        )
        
        # فاز ۱۴: بررسی Overfitting
        print(f"\n{'='*65}\n🟢 فاز ۱۴: بررسی تخصصی Overfitting\n{'='*65}")
        p14 = run_overfitting_analysis(
            p5,
            output_folder=f'phase14_overfitting_analysis_plots_{suffix}'
        )
        
        # فاز ۱۵: جدول مقایسه نهایی
        print(f"\n{'='*65}\n🟢 فاز ۱۵: جدول مقایسه نهایی جامع\n{'='*65}")
        p15 = run_master_table(
            p5,
            output_folder=f'phase15_master_table_{suffix}'
        )
        
        # فاز ۱۶: نمودارهای مقایسه‌ای نهایی
        print(f"\n{'='*65}\n🟢 فاز ۱۶: نمودارهای مقایسه‌ای نهایی\n{'='*65}")
        p16 = run_final_visualizations(
            p5, p15.master_df,
            output_folder=f'phase16_final_visualizations_{suffix}'
        )
        
        # فاز ۱۷: تحلیل خودکار و نتیجه‌گیری نهایی
        print(f"\n{'='*65}\n🟢 فاز ۱۷: تحلیل خودکار و نتیجه‌گیری نهایی\n{'='*65}")
        p17 = run_executive_summary(
            p5, p15,
            output_folder=f'phase17_executive_summary_{suffix}'
        )
        
        return df, corr, p2, p3, p4, p5, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17

    # ══════════════════════════════════════════════════════════
    #  اجرای پردازش برای هر دو مجموعه داده
    # ══════════════════════════════════════════════════════════
    with_fe_res = _process_dataset(
        df_with_fe,
        dataset_key=DATASET_KEY_WITH_FE,
        file_path=FILE_WITH_FE,
        suffix='with_fe',
        label='با مهندسی ویژگی'
    )
    
    no_fe_res = _process_dataset(
        df_no_fe,
        dataset_key=DATASET_KEY_NO_FE,
        file_path=FILE_NO_FE,
        suffix='no_fe',
        label='بدون مهندسی ویژگی'
    )
    
    # استخراج نتایج به صورت خوانا
    (df_with_fe, _, phase2_with_fe, phase3_with_fe, phase4_with_fe, phase5_with_fe,
     phase7_with_fe, phase8_with_fe, phase9_with_fe, phase10_with_fe, phase11_with_fe,
     phase12_with_fe, phase13_with_fe, phase14_with_fe, phase15_with_fe, phase16_with_fe,
     phase17_with_fe) = with_fe_res
     
    (df_no_fe, _, phase2_no_fe, phase3_no_fe, phase4_no_fe, phase5_no_fe,
     phase7_no_fe, phase8_no_fe, phase9_no_fe, phase10_no_fe, phase11_no_fe,
     phase12_no_fe, phase13_no_fe, phase14_no_fe, phase15_no_fe, phase16_no_fe,
     phase17_no_fe) = no_fe_res

    # ══════════════════════════════════════════════════════════
    #  بازگرداندن تمام نتایج
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("✅ فازهای ۰ تا ۱۷ با موفقیت کامل شدند.")
    print("=" * 65)

    return {
    'df_with_fe': df_with_fe,
    'phase2_with_fe': phase2_with_fe,
    'phase3_with_fe': phase3_with_fe,
    'phase4_with_fe': phase4_with_fe,
    'phase5_with_fe': phase5_with_fe,
    'phase7_with_fe': phase7_with_fe,
    'phase8_with_fe': phase8_with_fe,
    'phase9_with_fe': phase9_with_fe,
    'phase10_with_fe': phase10_with_fe,
    'phase11_with_fe': phase11_with_fe,
    'phase12_with_fe': phase12_with_fe,
    'phase13_with_fe': phase13_with_fe,
    'phase14_with_fe': phase14_with_fe,
    'phase15_with_fe': phase15_with_fe,
    'phase16_with_fe': phase16_with_fe,
    'phase17_with_fe': phase17_with_fe,
    'df_no_fe': df_no_fe,
    'phase2_no_fe': phase2_no_fe,
    'phase3_no_fe': phase3_no_fe,
    'phase4_no_fe': phase4_no_fe,
    'phase5_no_fe': phase5_no_fe,
    'phase7_no_fe': phase7_no_fe,
    'phase8_no_fe': phase8_no_fe,
    'phase9_no_fe': phase9_no_fe,
    'phase10_no_fe': phase10_no_fe,
    'phase11_no_fe': phase11_no_fe,
    'phase12_no_fe': phase12_no_fe,
    'phase13_no_fe': phase13_no_fe,
    'phase14_no_fe': phase14_no_fe,
    'phase15_no_fe': phase15_no_fe,
    'phase16_no_fe': phase16_no_fe,
    'phase17_no_fe': phase17_no_fe,
}
# ══════════════════════════════════════════════════════════════
#  اجرای اصلی برنامه
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── مرحله ۱: پیش‌پردازش (در صورت نیاز کامنت را بردارید) ──
    print("\n" + "=" * 65)
    print("📍 مرحله ۱: اجرای پایپ‌لاین پیش‌پردازش")
    print("=" * 65)
    run_preprocessing_pipeline()

    # ── مرحله ۲: فازهای مدل‌سازی ────────────────────────────
    print("\n" + "=" * 65)
    print("📍 مرحله ۲: اجرای فازهای مدل‌سازی (۰ تا ۱۷)")
    print("=" * 65)
    
    results = run_modeling_phases()

    # ── خلاصه نهایی هوشمند ──────────────────────────────────
    print("\n" + "=" * 65)
    print("🏆 خلاصه نهایی پروژه")
    print("=" * 65)
    
    for suffix, label in [('_with_fe', 'با مهندسی ویژگی'), ('_no_fe', 'بدون مهندسی ویژگی')]:
        p17 = results.get(f'phase17{suffix}')
        if p17:
            print(f"\n🟢 مجموعه داده {label}:")
            print(f"   • مدل برتر نهایی : {p17.overall_best_model}")
            print(f"   • Test R²        : {p17.overall_best_r2:.4f}")
            print(f"   • گزارش کامل     : executive_summary_report.txt")
            
    print("\n" + "=" * 65)
    print("🎉 تمام مراحل با موفقیت کامل شدند!")
    print("=" * 65)