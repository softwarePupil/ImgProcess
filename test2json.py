import os
import json
from PIL import Image
from tqdm import tqdm

def get_bbox(line):
    box = []
    # 去除行尾的换行符
    line = line.strip()
    # 拆分行并选择所需的值
    parts = line.split()
    values = [float(parts[1])] + [int(x) for x in parts[2:]]
    box.append(values[1])
    box.append(values[2])
    box.append(values[3]-values[1])
    box.append(values[4]-values[2])
    return values[0], box

def write_image_name(image_name):
    image_path = os.path.join("./VOCdevkit/VOC2007/JPEGImages/",image_name)
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        path, extension = os.path.splitext(image_path)
        new_image_path = path + ".bmp"
    try:
        image = Image.open(new_image_path)
    except FileNotFoundError:
        path, extension = os.path.splitext(image_path)
        new_image_path = path + ".jpg"
    image = Image.open(new_image_path)
    
    return new_image_path.split("/")[-1]

def traverse_files_in_directory(path):
    upload_json = []
    for root, directories, files in os.walk(path):
        for file_name in tqdm(files):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if line:
                    dt = {}
                    dt['name'] = write_image_name(file_name)
                    dt['category_id'] = 1
                    score, box = get_bbox(line)
                    dt['bbox'] = box
                    dt['score'] = score
                    upload_json.append(dt)
            
    return upload_json
            

# 指定路径
directory_path = "./map_out/detection-results"

# 遍历路径下的所有文件
data = traverse_files_in_directory(directory_path)
with open("./results.json", "w") as file:
    json.dump(data, file, indent=4)