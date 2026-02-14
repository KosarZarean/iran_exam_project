#!/usr/bin/env python3
"""
تست ساختار پروژه
"""

import os

def check_structure():
    """بررسی ساختار پروژه"""
    base_dir = os.path.dirname(__file__)
    
    required_dirs = ['data', 'src', 'results', 'models']
    required_files = ['requirements.txt', 'README.md', 'run_experiment.py']
    
    print("🔍 بررسی ساختار پروژه...")
    
    # بررسی پوشه‌ها
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ پوشه '{dir_name}' وجود دارد")
        else:
            print(f"❌ پوشه '{dir_name}' یافت نشد")
    
    # بررسی فایل‌ها
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            print(f"✅ فایل '{file_name}' وجود دارد")
        else:
            print(f"❌ فایل '{file_name}' یافت نشد")
    
    # بررسی فایل داده
    data_file = os.path.join(base_dir, 'data', 'iran_exam.csv')
    if os.path.exists(data_file):
        print(f"✅ فایل داده 'iran_exam.csv' وجود دارد")
    else:
        print(f"⚠️ فایل داده 'iran_exam.csv' یافت نشد")
    
    print("\n🎯 ساختار پروژه بررسی شد")

if __name__ == '__main__':
    check_structure()