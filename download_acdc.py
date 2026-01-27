#!/usr/bin/env python
"""
Script tải dữ liệu ACDC từ Google Drive
"""
import os
import subprocess
import sys

# Cài gdown nếu chưa có
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    import gdown

# Đường dẫn
PROJECT_DIR = "/teamspace/studios/this_studio"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "ACDC")
ZIP_FILE = os.path.join(PROJECT_DIR, "ACDC.zip")

# Google Drive file ID cho ACDC
FILE_ID = "16b2paVzOwlk3UOGlDyYsDWmP0ZDza-83"

print("=" * 50)
print("Tải dữ liệu ACDC từ Google Drive")
print("=" * 50)

# 1. Tải file
print(f"\n[1/3] Đang tải ACDC.zip...")
url = f"https://drive.google.com/uc?id={FILE_ID}"
gdown.download(url, ZIP_FILE, quiet=False)

# 2. Tạo thư mục đích
print(f"\n[2/3] Tạo thư mục {DATA_DIR}...")
os.makedirs(DATA_DIR, exist_ok=True)

# 3. Giải nén
print(f"\n[3/3] Đang giải nén vào {DATA_DIR}...")
import zipfile
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

# 4. Xóa file zip
print(f"\n[4/4] Dọn dẹp file zip...")
os.remove(ZIP_FILE)

print("\n" + "=" * 50)
print("Hoàn tất! Dữ liệu ACDC đã được tải vào:")
print(f"  {DATA_DIR}")
print("=" * 50)

# Liệt kê nội dung
print("\nNội dung thư mục:")
for item in os.listdir(DATA_DIR):
    item_path = os.path.join(DATA_DIR, item)
    if os.path.isdir(item_path):
        print(f"  📁 {item}/")
    else:
        print(f"  📄 {item}")
