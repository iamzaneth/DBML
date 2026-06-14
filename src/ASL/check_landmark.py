from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# Nếu muốn kiểm tra toàn bộ label: dùng thư mục gốc
root = Path("data/processed/ASL/bless")

# Nếu chỉ muốn kiểm tra 1 label, đổi thành:
# root = Path("data/processed/ASL/landmarks/a")

files = sorted(root.rglob("*.npz"))

print("=" * 80)
print("NPZ CHECKER")
print("=" * 80)
print("Root folder:", root)
print("Total npz files:", len(files))

if not files:
    raise SystemExit("Không tìm thấy file .npz nào.")


def is_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


def print_array_info(name, arr, max_values=20):
    print(f"\n  Key: {name}")
    print(f"    Shape : {arr.shape}")
    print(f"    Dtype : {arr.dtype}")
    print(f"    Size  : {arr.size}")

    if arr.size == 0:
        print("    Empty array")
        return

    if is_numeric(arr):
        finite_mask = np.isfinite(arr)
        nan_count = np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
        inf_count = np.isinf(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0

        print(f"    NaN   : {nan_count}")
        print(f"    Inf   : {inf_count}")

        if finite_mask.any():
            finite_values = arr[finite_mask]
            print(f"    Min   : {finite_values.min()}")
            print(f"    Max   : {finite_values.max()}")
            print(f"    Mean  : {finite_values.mean()}")
            print(f"    Std   : {finite_values.std()}")
        else:
            print("    Không có giá trị finite để tính min/max/mean/std")

        flat = arr.reshape(-1)
        print(f"    First {max_values} values:")
        print("   ", flat[:max_values])

        if arr.ndim >= 2:
            print(f"    First frame first {max_values} values:")
            print("   ", arr[0].reshape(-1)[:max_values])

    else:
        flat = arr.reshape(-1)
        print(f"    First {max_values} values:")
        print("   ", flat[:max_values])


# =========================
# 1. Kiểm tra chi tiết file đầu tiên
# =========================

first_file = files[0]

print("\n" + "=" * 80)
print("DETAIL OF FIRST FILE")
print("=" * 80)
print("File:", first_file)

with np.load(first_file, allow_pickle=True) as data:
    print("Keys:", data.files)

    for key in data.files:
        arr = data[key]
        print_array_info(key, arr)


# =========================
# 2. Kiểm tra toàn bộ file
# =========================

print("\n" + "=" * 80)
print("SUMMARY ALL NPZ FILES")
print("=" * 80)

key_patterns = Counter()
shape_counter = defaultdict(Counter)
dtype_counter = defaultdict(Counter)
label_counter = Counter()

error_files = []
nan_files = []
inf_files = []
empty_files = []

for file in files:
    try:
        label_counter[file.parent.name] += 1

        with np.load(file, allow_pickle=True) as data:
            keys = tuple(sorted(data.files))
            key_patterns[keys] += 1

            if len(data.files) == 0:
                empty_files.append(file)

            for key in data.files:
                arr = data[key]

                shape_counter[key][arr.shape] += 1
                dtype_counter[key][str(arr.dtype)] += 1

                if arr.size == 0:
                    empty_files.append(file)

                if is_numeric(arr):
                    if np.issubdtype(arr.dtype, np.floating):
                        if np.isnan(arr).any():
                            nan_files.append((file, key))
                        if np.isinf(arr).any():
                            inf_files.append((file, key))

    except Exception as e:
        error_files.append((file, str(e)))


print("\nFiles by label/folder:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count}")

print("\nKey patterns:")
for keys, count in key_patterns.most_common():
    print(f"  {count} files -> {keys}")

print("\nShapes by key:")
for key, counter in shape_counter.items():
    print(f"\n  Key: {key}")
    for shape, count in counter.most_common():
        print(f"    {count} files -> shape {shape}")

print("\nDtypes by key:")
for key, counter in dtype_counter.items():
    print(f"\n  Key: {key}")
    for dtype, count in counter.most_common():
        print(f"    {count} files -> dtype {dtype}")


# =========================
# 3. Cảnh báo lỗi dữ liệu
# =========================

print("\n" + "=" * 80)
print("WARNINGS")
print("=" * 80)

print("Error files:", len(error_files))
for file, err in error_files[:20]:
    print(" ", file, "->", err)

print("\nEmpty files/arrays:", len(empty_files))
for file in empty_files[:20]:
    print(" ", file)

print("\nFiles containing NaN:", len(nan_files))
for file, key in nan_files[:20]:
    print(" ", file, "| key:", key)

print("\nFiles containing Inf:", len(inf_files))
for file, key in inf_files[:20]:
    print(" ", file, "| key:", key)


# =========================
# 4. Kiểm tra đồng nhất cấu trúc
# =========================

print("\n" + "=" * 80)
print("CONSISTENCY CHECK")
print("=" * 80)

if len(key_patterns) == 1:
    print("OK: Tất cả file có cùng danh sách keys.")
else:
    print("WARNING: Các file có danh sách keys khác nhau.")

for key, counter in shape_counter.items():
    if len(counter) == 1:
        print(f"OK: Key '{key}' có shape đồng nhất.")
    else:
        print(f"WARNING: Key '{key}' có nhiều shape khác nhau:")
        for shape, count in counter.most_common():
            print(f"  {count} files -> {shape}")

for key, counter in dtype_counter.items():
    if len(counter) == 1:
        print(f"OK: Key '{key}' có dtype đồng nhất.")
    else:
        print(f"WARNING: Key '{key}' có nhiều dtype khác nhau:")
        for dtype, count in counter.most_common():
            print(f"  {count} files -> {dtype}")

print("\nDone.")