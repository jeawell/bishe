from PIL import Image
import numpy as np

def read_png_to_numpy(file_path):
    # 使用 PIL 打开 PNG 图像
    image = Image.open(file_path).convert("RGBA")  # 支持带 alpha 通道的 PNG

    # 将图像转换为 numpy 数组
    image_array = np.array(image)

    return image_array

# 示例：读取图片并输出其 shape 和像素值范围
if __name__ == "__main__":
    path = r"G:\WeChat Files\WeChat Files\wxid_qbmwuwan9yd122\FileStorage\File\2025-05\fpc示例(1)\fpc示例\_img8_6_9_192610-露铜.png"  # 替换为你的 PNG 文件路径
    img_array = read_png_to_numpy(path)
    print("图像形状 (height, width, channels):", img_array.shape)
    print("图像数据类型:", img_array.dtype)
    print("像素值范围: min =", img_array.min(), ", max =", img_array.max())
