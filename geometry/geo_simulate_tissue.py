import numpy as np
import h5py
import matplotlib.pyplot as plt


def generate_geometry(shape=(256, 256, 5)):
    data = np.zeros(shape, dtype=np.float64)

    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    xx, yy = np.meshgrid(x, y)

    # 各组织区域定义（保持原有逻辑不变）
    r_csf = 0.3
    mask_csf = (xx ** 2 + yy ** 2 <= r_csf ** 2)
    data[mask_csf, 0] = 1.0  # PD

    r_wm_outer = 0.5
    mask_wm = (xx ** 2 + yy ** 2 > r_csf ** 2) & (xx ** 2 + yy ** 2 <= r_wm_outer ** 2)
    data[mask_wm, 0] = 0.7

    mask_gm = (xx > -0.7) & (xx < -0.3) & (yy > -0.4) & (yy < 0.4)
    data[mask_gm, 0] = 0.85

    mask_bone = (xx >= 0.6) & (yy <= -0.6) & (xx + yy <= 0.0)
    data[mask_bone, 0] = 0.3

    mask_lesion = (xx > -0.05) & (xx < 0.05) & (yy > -0.05) & (yy < 0.05)
    data[mask_lesion, 0] = 0.9

    # T1/T2参数设置（保持原有组织赋值逻辑不变）
    data[..., 1] = 800  # 白质默认
    data[mask_csf, 1] = 4000
    data[mask_gm, 1] = 1200
    data[mask_bone, 1] = 250
    data[mask_lesion, 1] = 1500

    data[..., 2] = 80  # 白质默认
    data[mask_csf, 2] = 2000
    data[mask_gm, 2] = 100
    data[mask_bone, 2] = 0.5
    data[mask_lesion, 2] = 150

    data[..., 3] = 2.0 * xx * yy
    data[..., 4] = 0.8e-3
    data[mask_csf, 4] = 3.0e-3

    # 新增：单独处理背景（将背景区域的T1和T2设为NaN）
    background_mask = ~(mask_csf | mask_wm | mask_gm | mask_bone | mask_lesion)
    data[background_mask, 1] = np.nan  # 背景T1设为NaN
    data[background_mask, 2] = np.nan  # 背景T2设为NaN

    masks = {
        "mask_csf": mask_csf,
        "mask_wm": mask_wm,
        "mask_gm": mask_gm,
        "mask_bone": mask_bone,
        "mask_lesion": mask_lesion,
        "background_mask": background_mask
    }
    return data, masks


def generate_metadata():
    offset = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    resolution = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
    return offset, resolution


def save_h5_file(filename, data, offset, resolution):
    with h5py.File(filename, "w") as f:
        sample = f.create_group("sample")
        sample.create_dataset("data", data=data.astype(np.float64), dtype=np.float64)
        sample.create_dataset("offset", data=offset, dtype=np.float64)
        sample.create_dataset("resolution", data=resolution, dtype=np.float64)


if __name__ == "__main__":
    # 生成数据和掩膜
    data, masks = generate_geometry()

    # 生成元数据
    offset, resolution = generate_metadata()

    # 保存文件 (请修改为你的实际路径)
    save_path = "h5_path"
    save_h5_file(save_path, data, offset, resolution)

    # ==== 验证与可视化 ====
    print("验证指标：")
    print(f"PD范围：{np.min(data[..., 0]):.2f} - {np.max(data[..., 0]):.2f}")
    print(f"T1范围：{np.nanmin(data[..., 1]):.2f} ms - {np.nanmax(data[..., 1]):.2f} ms")
    print(f"T2范围：{np.nanmin(data[..., 2]):.4f} ms - {np.nanmax(data[..., 2]):.2f} ms")
    print(f"B0范围：{np.min(data[..., 3]):.2f} - {np.max(data[..., 3]):.2f} ppm")

    plt.figure(figsize=(15, 8))
    for i in range(5):
        plt.subplot(2, 3, i + 1)

        # 直接显示数据
        show_data = data[:, :, i]

        # 使用jet颜色映射，并设置vmin和vmax
        if i == 1:  # T1
            plt.imshow(show_data, cmap='jet', vmin=0, vmax=4000)  # T1范围：0-4000 ms
        elif i == 2:  # T2
            plt.imshow(show_data, cmap='jet', vmin=0, vmax=2000)  # T2范围：0-2000 ms
        else:
            plt.imshow(show_data, cmap='jet')

        plt.colorbar()

        # 添加标记
        plt.text(0.5, -0.1, f"({chr(97 + i)})", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14,
                 color='black')
        plt.axis('off')

    plt.tight_layout()
    plt.show()