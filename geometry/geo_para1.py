import numpy as np
import h5py
import matplotlib.pyplot as plt


# **生成几何形状的 3D 数据 (5层)**
def generate_geometry(shape=(216, 180, 5)):
    data = np.zeros(shape, dtype=np.float64)  # 必须使用双精度

    # ========================== 第0层：质子密度 PD ==========================
    # 圆形 (模拟脑脊液)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    circle = (xx ** 2 + yy ** 2 <= 1).astype(np.float64)
    data[20:70, 100:150, 0] = circle * 1.0  # PD=1.0

    # 矩形 (模拟灰质)
    data[100:150, 75:105, 0] = 0.6  # PD=0.6

    # 三角形 (模拟骨骼)
    triangle = np.triu(np.ones((50, 50))) * 0.3
    data[20:70, 20:70, 0] = triangle  # PD=0.3

    # ========================== 第1层：T1弛豫时间 ==========================
    data[..., 1] = 800  # 默认值 (白质)
    data[20:70, 100:150, 1] = 4000  # 脑脊液 T1=4000ms
    data[100:150, 75:105, 1] = 1200  # 灰质 T1=1200ms

    # ========================== 第2层：T2弛豫时间 ==========================
    data[..., 2] = 80  # 默认值 (白质)
    data[20:70, 100:150, 2] = 2000  # 脑脊液 T2=2000ms
    data[100:150, 75:105, 2] = 100  # 灰质 T2=100ms

    # ========================== 第3层：B0不均匀性 ==========================
    xx, yy = np.meshgrid(np.linspace(-1, 1, 180), np.linspace(-1, 1, 216))
    data[..., 3] = 2.0 * xx * yy  # 对角线梯度 (ppm)

    # ========================== 第4层：扩散系数 ==========================
    data[..., 4] = 0.8e-3  # 默认值
    data[20:70, 100:150, 4] = 3.0e-3  # 脑脊液扩散系数

    return data


# **元数据定义 (严格匹配示例格式)**
def generate_metadata():
    offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)  # 双精度 (3,1)
    resolution = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)  # 体素分辨率
    return offset, resolution


# **保存HDF5文件**
def save_h5_file(filename, data, offset, resolution):
    with h5py.File(filename, "w") as f:
        # 创建组结构
        sample = f.create_group("sample")

        # 保存数据集（严格匹配shape和dtype）
        sample.create_dataset("data", data=data.astype(np.float64), dtype=np.float64)  # 显式指定双精度
        sample.create_dataset("offset", data=offset, dtype=np.float64)
        sample.create_dataset("resolution", data=resolution, dtype=np.float64)


# **主程序**
if __name__ == "__main__":
    # 生成数据（注意shape为216x180x5）
    data = generate_geometry(shape=(216, 180, 5))

    # 生成元数据
    offset, resolution = generate_metadata()

    # 保存文件
    save_h5_file(
        "h5_path",
        data, offset, resolution
    )

    # 数值验证
    print("Data shape:", data.shape)  # 应输出 (216, 180, 5)
    print("PD range:", np.min(data[..., 0]), np.max(data[..., 0]))  # 应≈[0.3, 1.0]
    print("T1 range:", np.min(data[..., 1]), np.max(data[..., 1]))  # 应≈[800, 4000]
    print("T2 range:", np.min(data[..., 2]), np.max(data[..., 2]))  # 应≈[80, 2000]

    # 可视化验证
    plt.figure(figsize=(15, 3))
    titles = ["PD", "T1", "T2", "B0", "Diffusion"]
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(data[:, :, i], cmap="gray")
        plt.title(titles[i])
    plt.show()