"""
Create a Kaggle-compatible zip file with forward-slash paths.
PowerShell's Compress-Archive uses backslashes which Kaggle rejects.
This script uses Python's zipfile module which properly uses forward slashes.
"""
import os
import zipfile
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.zip")

# Folders to include in the zip
FOLDERS_TO_ZIP = ["separated_scans", "unified_dataset"]

def create_kaggle_zip():
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
        print(f"[*] Removed old {OUTPUT_ZIP}")

    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder_name in FOLDERS_TO_ZIP:
            folder_path = os.path.join(DATA_DIR, folder_name)
            if not os.path.exists(folder_path):
                print(f"[!] Skipping {folder_name} — not found at {folder_path}")
                continue

            print(f"[*] Adding {folder_name}/ ...")
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    # Create archive name relative to DATA_DIR with forward slashes
                    rel_path = os.path.relpath(abs_path, DATA_DIR)
                    # Convert backslashes to forward slashes for Kaggle/Linux compatibility
                    arc_name = rel_path.replace("\\", "/")
                    zf.write(abs_path, arc_name)
                    file_count += 1

                    if file_count % 500 == 0:
                        print(f"    ... {file_count} files added")

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\n[+] Done! Created {OUTPUT_ZIP}")
    print(f"    Total files: {file_count}")
    print(f"    Zip size: {zip_size_mb:.1f} MB")

    # Verify: list first 10 entries to confirm forward slashes
    print(f"\n[*] Verifying zip paths (first 10 entries):")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
        for i, name in enumerate(zf.namelist()[:10]):
            print(f"    {name}")
        if len(zf.namelist()) > 10:
            print(f"    ... and {len(zf.namelist()) - 10} more")

if __name__ == "__main__":
    create_kaggle_zip()
