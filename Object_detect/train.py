from roboflow import Roboflow

# Khởi tạo Roboflow với API key
rf = Roboflow(api_key="Kw3rtMLVahV7lmO48WV1")

# Tải dataset Earphone-1
project_earphone = rf.workspace("isods-xhfnh").project("earphone-pn7ld")
dataset_earphone = project_earphone.version(1).download("yolov8")

# Tải dataset cellphone-0aodn
project_cellphone = rf.workspace("d1156414").project("cellphone-0aodn")
dataset_cellphone = project_cellphone.version(1).download("yolov8")

# Tải dataset headphone-t8jet
project_headphone = rf.workspace("javier-n5kep").project("headphone-t8jet")
dataset_headphone = project_headphone.version(12).download("yolov8")

print("Tải dataset hoàn tất! Kiểm tra thư mục: Earphone-1, cellphone-0aodn, headphone-t8jet")








import os
import glob
import shutil

def remap_labels(input_folder, output_folder, old_class=0, new_class=0):
    os.makedirs(output_folder, exist_ok=True)
    for label_file in glob.glob(os.path.join(input_folder, '*.txt')):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        new_file = os.path.join(output_folder, os.path.basename(label_file))
        with open(new_file, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if parts and int(parts[0]) == old_class:
                    parts[0] = str(new_class)
                f.write(' '.join(parts) + '\n')

combined_path = "combined_cheating_detect"
os.makedirs(combined_path, exist_ok=True)
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(combined_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(combined_path, split, 'labels'), exist_ok=True)

# Cellphone (class 0)
for split in ['train', 'valid', 'test']:
    src_images = f'cellphone-1/{split}/images'
    src_labels = f'cellphone-1/{split}/labels'
    dst_images = os.path.join(combined_path, split, 'images')
    dst_labels = os.path.join(combined_path, split, 'labels')
    if os.path.exists(src_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
    if os.path.exists(src_labels):
        shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)

# Earphone (remap 0 thành 1)
for split in ['train', 'valid', 'test']:
    src_images = f'Earphone-1/{split}/images'
    src_labels = f'Earphone-1/{split}/labels'
    dst_images = os.path.join(combined_path, split, 'images')
    dst_labels = os.path.join(combined_path, split, 'labels')
    if os.path.exists(src_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
    if os.path.exists(src_labels):
        remap_labels(src_labels, dst_labels, old_class=0, new_class=1)

# Headphone (remap 0 thành 2, không có test)
for split in ['train', 'valid']:
    src_images = f'headphone-12/{split}/images'
    src_labels = f'headphone-12/{split}/labels'
    dst_images = os.path.join(combined_path, split, 'images')
    dst_labels = os.path.join(combined_path, split, 'labels')
    if os.path.exists(src_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
    if os.path.exists(src_labels):
        remap_labels(src_labels, dst_labels, old_class=0, new_class=2)

# Tạo data.yaml
import yaml
data_yaml = {
    'path': combined_path,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 3,
    'names': ['cellphone', 'earphone', 'headphone']
}
with open(os.path.join(combined_path, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f)

print("Gộp dataset hoàn tất! Thư mục:", combined_path)