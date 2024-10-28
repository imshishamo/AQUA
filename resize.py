import os
from PIL import Image

def resize_images(folder_path, target_width, target_height):
    # 獲取目標資料夾下的所有檔案
    files = os.listdir(folder_path)

    for file in files:
        # 確保是圖片檔案
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, file)
            
            # 打開圖片
            img = Image.open(file_path)
            
            # 調整大小
            resized_img = img.resize((target_width, target_height))
            
            # 替換無效字元並保存圖片
            new_file_name = file.replace('(', '_').replace(')', '_')  # 將括號替換為底線
            new_file_path = os.path.join(folder_path, new_file_name)
            resized_img.save(new_file_path)

            print(f"已調整大小並保存: {file}")

# 設定目標資料夾和目標大小
target_folder = r'C:\Users\user\Desktop\Sophia\bubble_data\evening2label'  # 更換為你的實際資料夾路徑
target_width = 840
target_height = 480

# 呼叫函數
resize_images(target_folder, target_width, target_height)
