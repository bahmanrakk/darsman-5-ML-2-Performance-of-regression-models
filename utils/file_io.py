import pandas as pd
from pathlib import Path


def read_csv_simple(filename: str) -> pd.DataFrame:
    """
    خواندن فایل CSV از پوشه 'data' در همان سطح پروژه
    
    پارامترها:
        filename: نام فایل CSV (مثلاً 'used_cars.csv')
    
    بازگشت:
        pd.DataFrame: داده‌های خوانده شده یا None در صورت خطا
    """
    try:
        # مسیر پروژه = والدِ والدِ این فایل
        # utils/file_io.py → utils → project
        project_path = Path(__file__).parent.parent
        csv_path = project_path / 'data' / filename
        
        df = pd.read_csv(csv_path, skip_blank_lines=False)
        
        print(f"✅ فایل {filename} با موفقیت خوانده شد")
        print(f"شکل داده: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ فایل {filename} یافت نشد!")
        print(f"مسیر جستجو: {csv_path}")
        return None
        
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return None


def save_csv_file(df: pd.DataFrame, filename: str, 
                  index: bool = False, subfolder: str = None) -> bool:
    """
    ذخیره دیتافریم به فایل CSV در پوشه data
    
    پارامترها:
        df: دیتافریم pandas
        filename: نام فایل خروجی
        index: آیا ایندکس ذخیره شود؟
        subfolder: نام زیرپوشه اختیاری درون 'data'
    
    بازگشت:
        bool: True در صورت موفقیت
    """
    try:
        project_path = Path(__file__).parent.parent
        data_dir = project_path / 'data'
        
        if subfolder:
            output_dir = data_dir / subfolder
        else:
            output_dir = data_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / filename
        
        df.to_csv(csv_path, index=index)
        
        print(f"✅ فایل {filename} با موفقیت ذخیره شد")
        print(f"📁 مسیر: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در ذخیره فایل: {e}")
        return False