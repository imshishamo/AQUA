import json
import numpy as np
from PIL import Image, ImageDraw
import os

def create_mask_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 取得影像路徑
    image_path = data.get('imagePath', '')

    # 如果有指定影像寬高，可以提取，否則使用預設值
    image_width = data.get('imageWidth', 848)
    image_height = data.get('imageHeight', 480)

    # 創建一個空白的遮罩圖 (灰度模式)
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    # 提取每個多邊形的點
    shapes = data.get('shapes', [])

    for shape in shapes:
        label = shape.get('label', '')
        points = shape.get('points', [])

        if label and points:
            # 將浮點數座標四捨五入為整數
            rounded_points = [(round(x), round(y)) for x, y in points]
            
            # 將點的座標連接起來，形成一個多邊形
            draw.polygon(rounded_points, outline=1, fill=1)

    # 將遮罩圖轉換為 NumPy 陣列
    mask_array = np.array(mask)

    return mask_array, image_path

# 假設 JSON 檔案名稱為 example.json
path = r"C:\Users\user\Desktop\Sophia\bubble_data\evening2label_done"
file = os.listdir(path)
for i in file:
    if i.endswith(".json"):
        json_file_path = path +"/"+i
        filename = i.rstrip(".json")
        # 創建遮罩圖
        mask_array, image_path = create_mask_from_json(json_file_path)

        # 保存遮罩圖
        output_path = path + '/mask/'+filename+'.png'
        Image.fromarray((mask_array * 255).astype(np.uint8)).save(output_path)

        print(f"遮罩圖已保存至 {output_path}，來自影像 {image_path}")
