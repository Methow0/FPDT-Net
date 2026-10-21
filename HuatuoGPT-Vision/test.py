import os
import csv
from cli import HuatuoChatbot

# 初始化模型
bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7B")

# 要遍历的文件夹路径
root_folder = '/home/rxk/YJ/FullNet/data/GlaS/images/train'
output_csv = './test.csv'

# 模型查询语句
query = 'What does the picture show?'

# 收集所有图片路径
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
image_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(root_folder)
    for file in files if file.lower().endswith(image_extensions)
]

# 写入CSV（caption在前，filename在后）
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Caption', 'filename'])

    for image_path in image_paths:
        try:
            output = bot.inference(query, [image_path])
            caption = str(output)

            # 去除[]和引号
            caption = caption.strip("[]'")

            # 写入一行
            writer.writerow([caption, image_path])
            print(f"✅ Processed: {image_path}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

print(f"\n✅ All captions saved to {output_csv}")
