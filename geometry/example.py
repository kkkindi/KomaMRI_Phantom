# 读取示例文件并打印结构
import h5py
import matplotlib.pyplot as plt

example_file = "./KomaMRI.jl-master/examples/2.phantoms/brain.h5"
with h5py.File(example_file, "r") as f:
    def print_structure(name, obj):
        print(f"{name}: {obj}")
    f.visititems(print_structure)

# sample: <HDF5 group "/sample" (3 members)> 一个组，组织数据
# sample/data: <HDF5 dataset "data": shape (216, 180, 5), type "<f8"> 数据集，形状表3D，如三维体积或多通道，64位浮点，5层切片？
# sample/offset: <HDF5 dataset "offset": shape (3, 1), type "<f8"> 三维坐标偏移量
# sample/resolution: <HDF5 dataset "resolution": shape (3, 1), type "<f8">  体素在三个维度上的分辨率

import h5py

# 打开 HDF5 文件
with h5py.File("./KomaMRI.jl-master/examples/2.phantoms/brain.h5", "r") as f:
    # 访问数据集
    data = f["/sample/data"][:]
    offset = f["/sample/offset"][:]
    resolution = f["/sample/resolution"][:]

    # 打印数据
    print("Data shape:", data.shape)
    print("Offset:", offset)
    print("Resolution:", resolution)

slice_index = 2  # 查看第 3 层切片（从 0 开始计数）

# 可视化单层切片
plt.imshow(data[:, :, slice_index], cmap="gray")
plt.colorbar()
plt.title(f"Slice {slice_index}")
plt.show()