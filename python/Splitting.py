import os
import shutil
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ORIGINAL_FILE = r"datasets\\observe_ds\Situation_3\\indore_03_telemetry_v1.csv"

# Directory where the copy + split files will be created
OUTPUT_DIR = r"C:\\C-V2X-OODA\datasets"

# Number of data rows in each split file
ROWS_PER_FILE = 2000

# ============================================================
# 1. CREATE A COPY OF THE ORIGINAL
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

original_name = os.path.basename(ORIGINAL_FILE)

copy_file = os.path.join(
    OUTPUT_DIR,
    "COPY_" + original_name
)

# Do not overwrite an existing copy
if os.path.exists(copy_file):
    raise FileExistsError(
        f"Copy already exists:\n{copy_file}\n"
        "Delete/rename it manually if you want to create a new copy."
    )

shutil.copy2(ORIGINAL_FILE, copy_file)

print("Original file:")
print(ORIGINAL_FILE)

print("\nUntouched copy created:")
print(copy_file)

# ============================================================
# 2. READ ONLY THE COPY
# ============================================================

df = pd.read_csv(copy_file)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# 3. SPLIT THE COPY
# ============================================================

split_dir = os.path.join(OUTPUT_DIR, "splits")
os.makedirs(split_dir, exist_ok=True)

# Remove previously generated split files only
# (does NOT touch the original or its copy)
for filename in os.listdir(split_dir):
    if filename.startswith("telemetry_part_") and filename.endswith(".csv"):
        os.remove(os.path.join(split_dir, filename))

total_rows = len(df)

part_number = 1

for start in range(0, total_rows, ROWS_PER_FILE):

    end = min(start + ROWS_PER_FILE, total_rows)

    part = df.iloc[start:end]

    output_file = os.path.join(
        split_dir,
        f"telemetry_part_{part_number:03d}.csv"
    )

    part.to_csv(
        output_file,
        index=False
    )

    print(
        f"Created {output_file} "
        f"-> rows {start + 1} to {end} "
        f"({len(part)} rows)"
    )

    part_number += 1

# ============================================================
# 4. FINAL CHECK
# ============================================================

number_of_parts = part_number - 1

print("\n============================================")
print("SPLITTING COMPLETE")
print("============================================")

print(f"Original       : {ORIGINAL_FILE}")
print(f"Working copy   : {copy_file}")
print(f"Split directory: {split_dir}")
print(f"Number of parts: {number_of_parts}")
print(f"Rows per part  : {ROWS_PER_FILE}")
print(f"Total rows     : {total_rows}")

print("\nIMPORTANT:")
print("The original CSV was NOT modified.")
print("All splitting was performed from the COPY.")