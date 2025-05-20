import numpy as np
import h5py
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt

# ======================= 配置参数 =======================
TARGET_SHAPE = (256, 256, 5)  # 目标数据维度
PARAM_LAYERS = {
    0: "PD",  # 质子密度
    1: "T1",  # T1弛豫时间（单位ms）
    2: "T2",  # T2弛豫时间（单位ms）
    3: "B0",  # 磁场不均匀性（ppm）
    4: "Diff"  # 扩散系数（mm²/s）
}


# ====================== 参数映射配置 ======================
def create_parametric_maps(data):
    """生成5层参数映射"""
    param_volume = np.zeros(TARGET_SHAPE, dtype=np.float64)

    # ----------------- Layer 0: PD -----------------
    # 标准化到0-1范围
    pd_layer = (data - np.min(data)) / (np.max(data) - np.min(data))
    param_volume[:, :, 0] = resize(pd_layer, TARGET_SHAPE[:2], order=1)

    # ----------------- Layer 1: T1 -----------------
    # 根据强度划分T1值（示例逻辑）
    t1_map = np.where(data < 100, 1/800,  # 白质
                      np.where(data < 200, 1/1200,  # 灰质
                               1/4000))  # CSF
    param_volume[:, :, 1] = resize(t1_map, TARGET_SHAPE[:2], order=0)  # 最近邻

    # ----------------- Layer 2: T2 -----------------
    t2_map = np.where(data < 100, 1/80,  # 白质
                      np.where(data < 200, 1/100,  # 灰质
                               1/2000))  # CSF
    param_volume[:, :, 2] = resize(t2_map, TARGET_SHAPE[:2], order=0)

    # ----------------- Layer 3: B0 -----------------
    xx, yy = np.meshgrid(np.linspace(-1, 1, TARGET_SHAPE[1]),
                         np.linspace(-1, 1, TARGET_SHAPE[0]))
    param_volume[:, :, 3] = 0.5 * (xx + yy)  # 线性不均匀性

    # ---------------- Layer 4: Diff ----------------
    param_volume[:, :, 4] = np.where(data < 100, 0.8e-3, 3.0e-3)

    return param_volume


# ====================== 数据处理流程 ======================
# 加载原始数据
img = nib.load('nii_path')
original_data = img.get_fdata()[:, :, 70:75]  # 选择中间5层

# 数据预处理
processed_data = np.zeros((256, 256, 5))  # 目标尺寸初始化

for z in range(5):
    # 关键修改点：先旋转 → 后resize
    rotated_slice = np.rot90(original_data[:, :, z], k=0, axes=(0, 1))
    resized_slice = resize(rotated_slice, (256, 256),
                          order=1,
                          preserve_range=True,
                          anti_aliasing=True)
    processed_data[:, :, z] = resized_slice

# 生成参数映射（后续代码保持不变）
param_volume = create_parametric_maps(np.mean(processed_data, axis=2))

# ====================== HDF5存储 ======================
with h5py.File("your_path", "w") as f:
    sample = f.create_group("sample")

    # 主数据集 (256,256,5)
    sample.create_dataset("data",
                          data=param_volume.transpose(1, 0, 2),  # 修复维度顺序
                          dtype='f8',
                          compression="gzip")

    # 空间参数（严格匹配shape）
    sample.create_dataset("offset",
                          data=np.array([[0.0], [0.0], [0.0]], dtype='f8'))
    sample.create_dataset("resolution",
                          data=np.array([[1.0], [1.0], [5.0]], dtype='f8'))


# ====================== 验证输出 ======================
def validate_h5_structure(filename):
    with h5py.File(filename, "r") as f:
        print("\n文件结构验证:")
        print("sample组成员:", list(f["sample"].keys()))
        print("data shape:", f["sample/data"].shape)
        print("offset shape:", f["sample/offset"].shape)
        print("resolution shape:", f["sample/resolution"].shape)
        print("\n参数范围:")
        for i, name in PARAM_LAYERS.items():
            layer = f["sample/data"][:, :, i]
            print(f"{name}: {np.min(layer):.2f} ~ {np.max(layer):.2f}")


validate_h5_structure("your_path")

# ====================== 可视化 ======================

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(param_volume[:, :, i], cmap='jet')
    plt.colorbar()
    # 在每张图下面添加 (a) (b) 等标记
    plt.text(0.5, -0.1, f"({chr(97 + i)})", transform=plt.gca().transAxes,
             ha='center', va='center')
    plt.axis('off')

plt.tight_layout()
plt.show()