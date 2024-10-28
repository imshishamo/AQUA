from PIL import Image
import os

def binarize_images(folder_path, threshold=128):
    # 獲取目標資料夾下的所有檔案
    files = os.listdir(folder_path)

    for file in files:
        # 確保是圖片檔案（你也可以通過其他方法進行確認）
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, file)

            # 打開圖片
            img = Image.open(file_path)

            # 將圖片轉為灰度圖
            img_gray = img.convert('L')

            # 將灰度圖進行二值化
            img_binary = img_gray.point(lambda x: 0 if x < threshold else 255, '1')

            # 保存覆蓋原始圖片
            img_binary.save(file_path)

            print(f"已轉換為二值化並保存: {file}")

# 設定目標資料夾和二值化閾值（threshold）
target_folder = r"C:\Users\user\Desktop\Sophia\Pytorch-UNet-master\Pytorch-UNet-master\data\masks" # 更換為你的實際資料夾路徑
binary_threshold = 128  # 你可以調整這個閾值

# 調用函數
binarize_images(target_folder, binary_threshold)
