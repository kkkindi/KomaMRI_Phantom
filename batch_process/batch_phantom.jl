using KomaMRI
using KomaMRIFiles
using NIfTI  # 用于保存 NIfTI 格式的图像
using CUDA
using FileIO

sequence_path = "seq_path"  # 固定序列路径
phantom_folder = "phantom_path"  # 水模文件所在文件夹路径
output_image_dir = "results_path"  # 图像输出目录

# 创建输出目录
mkpath(output_image_dir)

# 定义扫描仪
sys = Scanner()

# 加载固定序列
seq = read_seq(sequence_path)

# 辅助函数用于重建
function reconstruct_2d_image(raw::RawAcquisitionData)
    acqData = AcquisitionData(raw)
    acqData.traj[1].circular = false #Removing circular window
    C = maximum(2*abs.(acqData.traj[1].nodes[:]))  #Normalize k-space to -.5 to .5 for NUFFT
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ C
    Nx, Ny = raw.params["reconSize"][1:2]
    recParams = Dict{Symbol,Any}()
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = true
    rec = reconstruction(acqData, recParams)
    image3d  = reshape(rec.data, Nx, Ny, :)
    image2d = (abs.(image3d) * prod(size(image3d)[1:2]))[:,:,1]
    return image2d
end

# 遍历文件夹中的所有 .phantom 文件
for (root, dirs, files) in walkdir(phantom_folder)
    for file in files
        if endswith(file, ".phantom")
            phantom_file = joinpath(root, file)

            # 导入水模
            obj = read_phantom(phantom_file)

            # 定义模拟参数并进行模拟
            sim_params = KomaMRICore.default_sim_params()
            raw = simulate(obj, seq, sys; sim_params)

            # 执行重建以获取图像
            image = reconstruct_2d_image(raw)

            # 使用 splitext 移除文件扩展名
            base_name_without_ext = first(splitext(basename(phantom_file)))

            # 保存为 .nii.gz 文件
            nii_filename = joinpath(output_image_dir, "$base_name_without_ext.nii")
            try
                nii_image = NIVolume(image)
                nii_image.header.qform_code = 1  # 设置 qform 代码
                nii_image.header.sform_code = 1  # 设置 sform 代码

                io = open(nii_filename, "w")
                NIfTI.write(io, nii_image)
                close(io)
                println("Successfully saved the NIfTI file: $nii_filename")
            catch e
                println("Error while saving the NIfTI file: $nii_filename")
                showerror(stdout, e)
            end
        end
    end
end
