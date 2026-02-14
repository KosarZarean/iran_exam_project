

#!/usr/bin/env python3
"""
فایل اجرای اصلی پروژه
"""

import sys
import os

# اضافه کردن مسیر src به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exam_main import main

if __name__ == '__main__':
    main()