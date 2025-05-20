import numpy as np
import h5py
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ======================= 配置参数 =======================
SPATIAL_SHAPE = (256, 256)  # 目标分辨率
RESOLUTION_MM = 1.0  # 1.0mm 各向同性分辨率
NUM_LAYERS = 5  # 生成 5 层数据


# ====================== 脑组织提取 ======================
def robust_brain_extraction(input_path, output_path):
    img = sitk.ReadImage(input_path)

    # 步骤0：数据类型转换
    if img.GetPixelID() in [sitk.sitkInt16, sitk.sitkUInt16]:
        img = sitk.Cast(img, sitk.sitkFloat32)

    # 步骤1：高斯平滑
    smooth_img = sitk.SmoothingRecursiveGaussian(img, sigma=1.0)

    # 步骤2：Otsu 阈值分割
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    otsu_img = otsu_filter.Execute(smooth_img)

    # 步骤3：生成边界距离场
    distance_map = sitk.SignedMaurerDistanceMap(otsu_img, squaredDistance=False)

    # 步骤4：定义物理边缘区域（5mm 缓冲）
    border_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    border_mask.CopyInformation(img)
    size = img.GetSize()
    spacing = img.GetSpacing()
    border_mm = 5.0  # 5mm 物理距离
    border_pixels = [int(border_mm / sp) for sp in spacing]

    # 标记物理边缘区域
    border_mask[0:border_pixels[0], :, :] = 1
    border_mask[size[0] - border_pixels[0]:, :, :] = 1
    border_mask[:, 0:border_pixels[1], :] = 1
    border_mask[:, size[1] - border_pixels[1]:, :] = 1
    border_mask[:, :, 0:border_pixels[2]] = 1
    border_mask[:, :, size[2] - border_pixels[2]:] = 1

    # 步骤5：联合运算（找到与边界相连的区域）
    connected_bg = sitk.BinaryThreshold(distance_map, -1e6, 0)  # 距离场负值区域
    final_bg = sitk.Or(connected_bg, border_mask)

    # 步骤6：生成最终掩膜
    final_mask = sitk.BinaryThreshold(distance_map, 0.1, 1e6)  # 正值区域（脑组织）
    final_mask = sitk.BinaryMorphologicalClosing(final_mask, [3, 3, 3])

    # 步骤7：去除小区域（保留最大连通区域）
    largest_cc = sitk.ConnectedComponent(final_mask)
    relabel = sitk.RelabelComponent(largest_cc, minimumObjectSize=5000)
    final_mask = sitk.Cast(relabel > 0, sitk.sitkUInt8)

    # 步骤8：应用掩膜
    result_img = sitk.Mask(img, final_mask)

    # 保存结果
    sitk.WriteImage(result_img, output_path)


# ====================== 参数映射配置 ======================
def create_parametric_maps(data_slice, mask):
    # 严格归一化到 [0.3, 1.0] 以增强对比度
    ρ = 0.3 + 0.7 * (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice))
    ρ[~mask] = 0.0

    # 动态分配 T1/T2（基于强度直方图分位数）
    q25 = np.percentile(data_slice[mask], 25)
    q75 = np.percentile(data_slice[mask], 75)

    # 白质（低强度区域）
    wm_mask = (data_slice >= q25) & (data_slice < q75 // 2)
    # 灰质（中强度区域）
    gm_mask = (data_slice >= q75 // 2) & (data_slice < q75)
    # CSF（高强度区域）
    csf_mask = (data_slice >= q75)

    T1 = np.zeros_like(data_slice)
    T1[wm_mask] = 0.8  # 白质 T1=800ms
    T1[gm_mask] = 1.4  # 灰质 T1=1400ms
    T1[csf_mask] = 3.5  # CSF T1=3500ms
    T1[~mask] = 1e6  # 背景 T1=inf

    T2 = np.zeros_like(data_slice)
    T2[wm_mask] = 0.07  # 白质 T2=70ms
    T2[gm_mask] = 0.1  # 灰质 T2=100ms
    T2[csf_mask] = 2.0  # CSF T2=2000ms
    T2[~mask] = 1e6  # 背景 T2=inf

    T2s = np.clip(T2 * 0.7, 0.03, 0.1)  # T2* 映射
    T2s[~mask] = 1e6

    # 频率偏移模拟（仅对组织生效）
    xx, yy = np.meshgrid(np.linspace(-1, 1, data_slice.shape[1]),
                         np.linspace(-1, 1, data_slice.shape[0]))
    Δw = 50 * np.exp(-(xx ** 2 + yy ** 2) / 0.5) * mask  # 增强磁场不均匀性

    return {"ρ": ρ, "T1": T1, "T2": T2, "T2s": T2s, "Δw": Δw}


# ====================== 主流程 ======================
def nii_to_phantom(input_nii, output_phantom):
    mask_path = "temp_mask.nii.gz"
    robust_brain_extraction(input_nii, mask_path)

    img, mask_img = nib.load(input_nii), nib.load(mask_path)
    data, mask = img.get_fdata(), mask_img.get_fdata() > 0
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # 选择中间 5 层
    z_center = data.shape[2] // 2
    data_slices = data[:, :, z_center - 2: z_center + 3]  # 5 层
    mask_slices = mask[:, :, z_center - 2: z_center + 3]

    # 处理每一层并拼接数据
    all_params = {key: [] for key in ["ρ", "T1", "T2", "T2s", "Δw"]}
    for z in range(5):
        data_slice = data_slices[:, :, z]
        mask_slice = mask_slices[:, :, z]

        # 关键修改：统一展平顺序（行优先）
        data_processed = resize(np.rot90(data_slice), (256, 256), order=1, preserve_range=True)
        mask_processed = resize(np.rot90(mask_slice.astype(float)), (256, 256), order=0) > 0

        # 按行优先展平（C-order）
        mask_flat = mask_processed.reshape(-1, order='C')
        params = create_parametric_maps(data_processed, mask_processed)

        for key in all_params:
            param_flat = params[key].reshape(-1, order='C')  # 按行展平
            all_params[key].append(param_flat)

    # 合并所有层的数据
    merged_params = {key: np.concatenate(all_params[key]) for key in all_params}

    # 计算坐标（按行优先展开）
    x_coords = np.linspace(-128, 128, 256) * 1e-3  # 单位：米
    y_coords = np.linspace(-128, 128, 256) * 1e-3
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    pos_x = np.tile(xx.reshape(-1, order='C'), 5)  # 按行展平后重复5层
    pos_y = np.tile(yy.reshape(-1, order='C'), 5)
    pos_z = np.repeat(np.arange(-2, 3) * 1e-3, 256 * 256)


    # 加载原始数据和 mask
    img = nib.load(input_nii).get_fdata()
    mask = nib.load(mask_path).get_fdata() > 0

    # 可视化中间层
    z_center = img.shape[2] // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, z_center], cmap='gray')
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, z_center], cmap='gray')
    plt.title("Mask")
    plt.show()

    with h5py.File(output_phantom, "w") as f:
        f.attrs.update({
            "Dims": 3,
            "Name": "5-Layer-Brain",
            "Ns": len(pos_x),  # 应等于 256*256*5 = 327680
            "Version": "0.9.0"
        })
        pos = f.create_group("position")
        pos.create_dataset("x", data=pos_x, dtype="f4")
        pos.create_dataset("y", data=pos_y, dtype="f4")
        pos.create_dataset("z", data=pos_z, dtype="f4")
        contrast = f.create_group("contrast")
        for key in ["ρ", "T1", "T2", "T2s", "Δw"]:
            contrast.create_dataset(key, data=merged_params[key], dtype="f4")

        ρ = f["contrast/ρ"][:]
        print("ρ 范围:", np.min(ρ[ρ > 0]), np.max(ρ[ρ > 0]))  # 应在 [0.2, 1.0] 之间
        T1 = f["contrast/T1"][:]
        T2 = f["contrast/T2"][:]
        print("组织区域参数示例:")
        print("T1:", T1[ρ > 0][:10])  # 应在 0.8-3.5 之间
        print("T2:", T2[ρ > 0][:10])  # 应在 0.07-2.0 之间


# ====================== 执行 ======================
nii_to_phantom(
    "nii.gz_path",
    "save_phantom"
)