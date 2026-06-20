"""
场状态可视化工具 —— 批量模式
读取 log/ 下所有 field_e*_t*.bin 文件，
每个文件生成一张三切片图 (z=16, 32, 48)。

用法：
    python viz_field.py                          # 处理 log/*.bin
    python viz_field.py path/to/fields/          # 处理指定目录
    python viz_field.py file.bin                 # 处理单个文件
"""

import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SZ = 64

def load_field(path):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != SZ * SZ * SZ:
        print(f"  ? {os.path.basename(path)}: size mismatch ({data.size}), skipping")
        return None
    return data.reshape(SZ, SZ, SZ)

def make_slices(path, out_dir="."):
    field = load_field(path)
    if field is None:
        return

    basename = os.path.splitext(os.path.basename(path))[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    for i, z in enumerate([16, 32, 48]):
        ax = axes[i]
        im = ax.imshow(field[:, :, z], cmap='viridis', aspect='equal')
        ax.set_title(f"z = {z}")
        ax.axis('off')

    fig.suptitle(basename, fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6)

    out_path = os.path.join(out_dir, f"{basename}.png")
    fig.savefig(out_path, dpi=150)
    print(f"  ? {out_path}")
    plt.close(fig)

def find_bin_files(target):
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        return sorted(glob.glob(os.path.join(target, "field_e*.bin")))
    return sorted(glob.glob(target))

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "log"

    files = find_bin_files(target)
    if not files:
        print(f"???: {target}")
        print("Usage: python viz_field.py [dir/file]")
        sys.exit(1)

    print(f"Found {len(files)} field file(s), generating...")
    for f in files:
        make_slices(f)
    print("Done.")
