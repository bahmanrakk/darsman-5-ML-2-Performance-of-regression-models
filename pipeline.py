import sys
from pathlib import Path

# ------------------------------------------------------------------
# اطمینان از اینکه مسیر پروژه در sys.path وجود دارد
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# import ترانسفورمرها
# ------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from transformers.nan_transformer import StandardizeNaNTransformer
from transformers.column_cleaner import CleanColumnNamesTransformer
from transformers.data_inspector import DataInspectorTransformer
from transformers.torque_processor import TorqueProcessorTransformer
from transformers.mileage_processor import MileageProcessorTransformer
from transformers.max_power_processor import MaxPowerProcessorTransformer
from transformers.car_age_processor import CarAgeProcessorTransformer
from transformers.engine_processor import EngineProcessorTransformer
from transformers.categorical_encoder import CategoricalEncoderTransformer
from transformers.column_dropper import ColumnDropperTransformer
from transformers.nan_analysis import NaNAnalysisTransformer
from transformers.drop_na_rows import DropNaRowsTransformer
from transformers.drop_duplicates import DropDuplicatesTransformer
from transformers.outlier_analyzer import OutlierAnalyzerTransformer
from transformers.outlier_clipper import OutlierClipperTransformer
from transformers.engine_encoder import EngineEncoderTransformer
from transformers.age_encoder import AgeEncoderTransformer
from transformers.seats_encoder import SeatsEncoderTransformer
from transformers.feature_engineering import FeatureEngineeringTransformer
from transformers.column_selector import ColumnSelectorTransformer
from transformers.multicollinearity_analyzer import MulticollinearityAnalyzerTransformer
from transformers.feature_normalizer import FeatureNormalizerTransformer  # ✨ جدید


def build_preprocessing_pipeline(
    drop_torque: bool = True,
    mileage_decimal_places: int = 4,
    max_power_decimal_places: int = 4,
    engine_decimal_places: int = 4,
    columns_to_drop: list = None,
    run_outlier_analysis: bool = False,
    final_columns: list = None,
    vif_analysis_columns: list = None,
    run_vif_analysis: bool = False,
    # ✨ پارامترهای جدید برای نرمال‌سازی
    sqrt_columns: list = None,
    log_columns: list = None,
    boxcox_columns: list = None,
    standardize_directly: list = None,
    run_normalization: bool = False
) -> Pipeline:
    
    if columns_to_drop is None:
        columns_to_drop = ['name']

    if final_columns is None:
        final_columns = [
            'km_driven', 'mileage', 'seats', 'age',
            'engine', 'max_power', 'torque_nm', 'torque_rpm',
            'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'fuel_CNG',
            'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark_Dealer',
            'transmission_bin', 'owner_ord',
            'selling_price'
        ]
    
    if vif_analysis_columns is None:
        vif_analysis_columns = []
                
    clipping_configs = [
        {'column': 'km_driven', 'min_thresh': 500, 'max_thresh': 300000},
        {'column': 'mileage', 'min_thresh': 8.0, 'max_thresh': 30.0},
        {'column': 'torque_nm', 'min_thresh': 100, 'max_thresh': 700},
        {'column': 'torque_rpm', 'min_thresh': 1200, 'max_thresh': 7000},
        {'column': 'max_power', 'min_thresh': 40, 'max_thresh': 250}
    ]

    pipeline = Pipeline(steps=[
        # مرحله 1: استانداردسازی مقادیر گمشده
        ('standardize_nan', StandardizeNaNTransformer()),
        
        # مرحله 2: پاک‌سازی نام ستون‌ها
        ('clean_columns', CleanColumnNamesTransformer()),
        
        # مرحله 3: بررسی و نمایش آمار اولیه
        ('inspect_data', DataInspectorTransformer()),
        
        # مرحله 4: پردازش کامل ستون torque
        ('process_torque', TorqueProcessorTransformer(
            drop_torque_column=drop_torque
        )),
        
        # مرحله 5: پردازش و استانداردسازی ستون mileage
        ('process_mileage', MileageProcessorTransformer(
            decimal_places=mileage_decimal_places
        )),
        
        # مرحله 6: پردازش و استانداردسازی ستون max_power
        ('process_max_power', MaxPowerProcessorTransformer(
            decimal_places=max_power_decimal_places
        )),
        
        # مرحله 7: محاسبه سن خودرو و حذف ستون year
        ('process_car_age', CarAgeProcessorTransformer()),
        
        # مرحله 8: پردازش و استانداردسازی ستون engine
        ('process_engine', EngineProcessorTransformer(
            decimal_places=engine_decimal_places
        )),
        
        # مرحله 9: رمزگذاری ویژگی‌های دسته‌ای
        ('encode_categorical', CategoricalEncoderTransformer()),
        
        # مرحله 10: حذف ستون‌های غیرضروری
        ('drop_columns', ColumnDropperTransformer(
            columns_to_drop=columns_to_drop
        )),
            
        # مرحله 11: تحلیل نهایی مقادیر گمشده
        ('analyze_nan', NaNAnalysisTransformer(verbose=True)),
        
        # مرحله 12: حذف ردیف‌های دارای مقادیر گمشده
        ('drop_na_rows', DropNaRowsTransformer(reset_index=True, verbose=True)),
        
        # مرحله 13: شناسایی و حذف ردیف‌های تکراری
        ('drop_duplicates', DropDuplicatesTransformer(subset=None, keep='first', remove_duplicates=True, verbose=True)),

        # مرحله 14: تحلیل داده‌های پرت
        ('analyze_outliers', OutlierAnalyzerTransformer(
            columns_to_analyze=[cfg['column'] for cfg in clipping_configs],
            generate_reports=run_outlier_analysis
        )),
        
        # مرحله 15: کلامپ کردن داده‌های پرت
        ('clip_outliers', OutlierClipperTransformer(clipping_configs=clipping_configs)),
        
        # مرحله 16: دسته‌بندی و کلامپ ستون engine
        ('encode_engine', EngineEncoderTransformer(column_name='engine')),
        
        # مرحله 17: دسته‌بندی و کلامپ ستون age
        ('encode_age', AgeEncoderTransformer(column_name='age', min_thresh=5, max_thresh=25)),
        
        # مرحله 18: دسته‌بندی و کلامپ ستون seats
        ('encode_seats', SeatsEncoderTransformer(column_name='seats')),

        # مرحله 19: مهندسی ویژگی
        ('feature_engineering', FeatureEngineeringTransformer()),
        
        # مرحله 20: تحلیل چندهمخطی‌بودگی (در صورت فعال بودن)
        ('analyze_multicollinearity', MulticollinearityAnalyzerTransformer(
            columns_to_analyze=vif_analysis_columns if run_vif_analysis else [],
            corr_threshold=0.75,
            vif_threshold=5.0,
            verbose=run_vif_analysis
        )),
        
        # مرحله 21: انتخاب ستون‌های نهایی
        ('select_columns', ColumnSelectorTransformer(columns_to_keep=final_columns)),
        
        # مرحله 22: نرمال‌سازی و تبدیل توزیع (در صورت فعال بودن) ✨ جدید
        ('normalize_features', FeatureNormalizerTransformer(
            sqrt_columns=sqrt_columns or [],
            log_columns=log_columns or [],
            boxcox_columns=boxcox_columns or [],
            standardize_directly=standardize_directly or [],
            keep_original=False,
            add_constant=1e-5
        )) if run_normalization else ('normalize_features', FeatureNormalizerTransformer()),
    ])
    
    return pipeline