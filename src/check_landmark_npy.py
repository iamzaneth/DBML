from pathlib import Path
import numpy as np

folder = Path("data/processed/landmarks/a")
files = sorted(folder.glob("*.npy"))

print("Total npy files:", len(files))

if files:
    arr = np.load(files[0])
    print("File:", files[0])
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("Min:", arr.min())
    print("Max:", arr.max())
    print("First frame first 20 values:")
    print(arr[0][:20])