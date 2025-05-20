import h5py
import numpy as np

def analyze_phantom_file(filepath):
    """分析 .phantom 文件的结构和内容"""
    with h5py.File(filepath, "r") as f:
        print("===== 根目录属性 =====")
        for attr_name in f.attrs:
            print(f"属性名: {attr_name}, 值: {f.attrs[attr_name]}")

        print("\n===== 文件结构 =====")
        _print_group_structure(f, indent="")

def _print_group_structure(group, indent):
    """递归打印组的结构"""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{indent}组: {key}")
            # 打印组的属性
            if item.attrs:
                print(f"{indent}  ├─ 属性:")
                for attr_name in item.attrs:
                    print(f"{indent}  │   {attr_name}: {item.attrs[attr_name]}")
            # 递归打印子组
            _print_group_structure(item, indent + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}数据集: {key}")
            print(f"{indent}  ├─ 形状: {item.shape}")
            print(f"{indent}  ├─ 数据类型: {item.dtype}")
            # 打印数据集的前几个值（示例）
            if len(item.shape) > 0 and item.size > 0:
                sample_data = item[:3] if item.size > 3 else item[:]
                print(f"{indent}  └─ 示例值: {sample_data}")

# 示例调用
if __name__ == "__main__":
    phantom_file = "example_phantom_path"  # 替换为你的 .phantom 文件路径
    analyze_phantom_file(phantom_file)