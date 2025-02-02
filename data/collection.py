import os
import gdown
import zipfile

url = "https://drive.google.com/uc?id=1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck"
output_zip = "sketchy_database.zip"
extract_dir = os.path.join('sketch-warp/data', 'sketchy_database')


def download_dataset(url, output_zip):
    print("Downloading dataset...")
    gdown.download(url, output_zip, quiet=False)
    print(f"Downloaded dataset to {output_zip}")


def extract_dataset(zip_file, extract_dir):
    print("Extracting dataset...")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted dataset to {extract_dir}")

download_dataset(url, output_zip)
extract_dataset(output_zip, extract_dir)

os.remove(output_zip)
print(f"Removed {output_zip} after extraction.")