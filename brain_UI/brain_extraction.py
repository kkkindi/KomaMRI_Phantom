import SimpleITK as sitk
import matplotlib.pyplot as plt

def robust_brain_extraction(input_path, output_path):
    """鲁棒的脑组织提取方案（处理多种数据类型）"""
    # 加载图像并转换数据类型
    img = sitk.ReadImage(input_path)

    # 步骤0：数据类型转换（关键修复）
    if img.GetPixelID() in [sitk.sitkInt16, sitk.sitkUInt16]:
        img = sitk.Cast(img, sitk.sitkFloat32)

    # 步骤1：预处理（兼容不同数据类型）
    # --------------------------------------------------
    # 高斯平滑（替代各向异性扩散）
    smooth_img = sitk.SmoothingRecursiveGaussian(img, sigma=1.0)

    # 步骤2：智能边缘检测
    # --------------------------------------------------
    # Otsu阈值预分割
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    otsu_img = otsu_filter.Execute(smooth_img)

    # 生成边界距离场
    distance_map = sitk.SignedMaurerDistanceMap(otsu_img, squaredDistance=False)

    # 步骤3：精确背景去除
    # --------------------------------------------------
    # 定义物理边缘区域（5mm缓冲）
    border_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    border_mask.CopyInformation(img)

    # 标记物理边缘区域
    size = img.GetSize()
    spacing = img.GetSpacing()

    # 计算物理距离对应的像素数
    border_mm = 5.0  # 5mm物理距离
    border_pixels = [int(border_mm / sp) for sp in spacing]

    # X方向
    border_mask[0:border_pixels[0], :, :] = 1
    border_mask[size[0] - border_pixels[0]:, :, :] = 1
    # Y方向
    border_mask[:, 0:border_pixels[1], :] = 1
    border_mask[:, size[1] - border_pixels[1]:, :] = 1
    # Z方向
    border_mask[:, :, 0:border_pixels[2]] = 1
    border_mask[:, :, size[2] - border_pixels[2]:] = 1

    # 联合运算：找到与边界相连的区域
    connected_bg = sitk.BinaryThreshold(distance_map, -1e6, 0)  # 距离场负值区域
    final_bg = sitk.Or(connected_bg, border_mask)

    # 生成最终掩膜
    final_mask = sitk.BinaryThreshold(distance_map, 0.1, 1e6)  # 正值区域（脑组织）
    final_mask = sitk.BinaryMorphologicalClosing(final_mask, [3, 3, 3])

    # 步骤4：应用掩膜
    result_img = sitk.Mask(img, final_mask)

    # 保存结果
    sitk.WriteImage(result_img, output_path)

    # 可视化验证
    slice_idx = img.GetSize()[2] // 2
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(sitk.GetArrayFromImage(img)[slice_idx], cmap='gray')
    plt.title('原始切片')

    plt.subplot(132)
    plt.imshow(sitk.GetArrayFromImage(final_mask)[slice_idx], cmap='gray')
    plt.title('最终掩膜')

    plt.subplot(133)
    plt.imshow(sitk.GetArrayFromImage(result_img)[slice_idx], cmap='gray')
    plt.title('处理结果')

    plt.show()


# 使用示例
if __name__ == "__main__":
    input_path = "your_path"
    output_path = "your_path"
    robust_brain_extraction(input_path, output_path)
