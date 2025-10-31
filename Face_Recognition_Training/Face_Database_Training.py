#!/usr/bin/env python
# coding: utf-8

# # 🗄️ Face Database Training Notebook
# 
# Notebook này hướng dẫn tạo Face Database bằng InsightFace (đa khuôn mặt, nhanh).
# 
# ## 📋 Các bước
# 1. Import thư viện, kiểm tra dataset
# 2. Khởi tạo InsightFace
# 3. Tạo face embeddings cho từng người trong `FaceDataset/train/`
# 4. Lưu `models/face_database_kaggle.pkl`
# 5. Test nhanh bằng ảnh trong `FaceDataset/val/`
# 

# ## 📦 Import và kiểm tra dataset
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import os, sys, time, pickle
import numpy as np
import cv2
from pathlib import Path

print("🔍 Checking dataset structure...")
train_dir = "FaceDataset/train/"
val_dir = "FaceDataset/val/"

for d in [train_dir, val_dir]:
    if not os.path.exists(d):
        print(f"❌ Missing: {d}")
    else:
        persons = [p for p in os.listdir(d) if os.path.isdir(os.path.join(d,p))]
        print(f"✅ {d} | people: {len(persons)}")
        for p in persons:
            imgs = [f for f in os.listdir(os.path.join(d,p)) if f.lower().endswith((".jpg",".jpeg",".png"))]
            print(f"   - {p}: {len(imgs)} images")


# ## 👁️‍🗨️ Khởi tạo InsightFace
# 

# In[3]:


import insightface
from numpy.linalg import norm

print("🔧 Initializing InsightFace (buffalo_l, CPU)...")
fa = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1)
print("✅ InsightFace ready!")


# ## 🗄️ Tạo Face Database và lưu models/face_database_kaggle.pkl
# 

# In[4]:


os.makedirs("models", exist_ok=True)
output_pkl = "models/face_database_kaggle.pkl"

database = {}
print("🚀 Building database from:", train_dir)

for person in sorted([p for p in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,p))]):
    person_dir = os.path.join(train_dir, person)
    embeddings = []
    imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    print(f"\n👤 {person}: {len(imgs)} images")
    for img_name in imgs:
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = fa.get(img)
        if len(faces) == 0:
            print(f"   ⚠️  {img_name}: no face detected")
            continue
        embeddings.append(faces[0].embedding)
        print(f"   ✅ {img_name}")
    if embeddings:
        database[person] = np.mean(embeddings, axis=0)
        print(f"   📊 saved mean embedding ({len(embeddings)} valid)")

with open(output_pkl, 'wb') as f:
    pickle.dump(database, f)

print("\n💾 Saved:", output_pkl)
print("👥 People in DB:", len(database))


# ## 🧪 Test nhanh trên FaceDataset/val
# 

# In[5]:


# Load DB
with open(output_pkl, 'rb') as f:
    db = pickle.load(f)

print("✅ DB loaded. People:", len(db))

# Chọn 1 ảnh từ val để test
tested = False
for person in [p for p in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir,p))]:
    imgs = [f for f in os.listdir(os.path.join(val_dir, person)) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not imgs:
        continue
    img_path = os.path.join(val_dir, person, imgs[0])
    print("\n🔍 Testing:", img_path)
    img = cv2.imread(img_path)
    faces = fa.get(img)
    if len(faces) == 0:
        print("⚠️  No face detected")
        continue
    query = faces[0].embedding
    # cosine similarity
    query = query / norm(query)
    best_score = -1
    best_name = "Unknown"
    for name, emb in db.items():
        e = emb / norm(emb)
        score = float(np.dot(query, e))
        if score > best_score:
            best_score = score
            best_name = name
    print(f"🎯 Best match: {best_name} | score: {best_score:.4f} | gt: {person}")
    tested = True
    break

if not tested:
    print("⚠️  No validation images found.")


# ## 📦 Import các thư viện cần thiết
# 

# In[6]:


import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime
import numpy as np
import pickle
from pathlib import Path

# Computer Vision
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Face Recognition
import insightface
from numpy.linalg import norm

print("✅ All libraries imported successfully!")


# ## 🔧 Khởi tạo InsightFace Model
# 

# In[7]:


print("🔧 Initializing InsightFace model...")

# Load InsightFace model
# Sử dụng buffalo_l model (tốt nhất cho face recognition)
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

print("✅ InsightFace model loaded successfully!")
print(f"📊 Model info:")
print(f"   Name: buffalo_l")
print(f"   Provider: CPU")
print(f"   Context ID: -1 (CPU)")

# Test model với một ảnh mẫu
print("\n🧪 Testing model...")
test_image_path = "FaceDataset/train/messi/1.jpg"  # Thay đổi đường dẫn nếu cần

if os.path.exists(test_image_path):
    img = cv2.imread(test_image_path)
    faces = model.get(img)
    print(f"✅ Test successful! Found {len(faces)} face(s) in test image")
    if len(faces) > 0:
        print(f"   Face embedding dimension: {faces[0].embedding.shape}")
else:
    print("⚠️  Test image not found, but model is ready to use")


# ## 📁 Kiểm tra Dataset Structure
# 

# In[8]:


# Đường dẫn dataset
dataset_train = "FaceDataset/train/"
dataset_val = "FaceDataset/val/"

print("🔍 Checking dataset structure...")
print("=" * 50)

# Kiểm tra thư mục train
if os.path.exists(dataset_train):
    print(f"✅ Training dataset found: {dataset_train}")
    
    # Đếm số người và ảnh
    people_count = 0
    total_images = 0
    
    for person_name in os.listdir(dataset_train):
        person_dir = os.path.join(dataset_train, person_name)
        if os.path.isdir(person_dir):
            people_count += 1
            images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            total_images += image_count
            print(f"   👤 {person_name}: {image_count} images")
    
    print(f"\n📊 Training Summary:")
    print(f"   👥 Total people: {people_count}")
    print(f"   🖼️  Total images: {total_images}")
else:
    print(f"❌ Training dataset not found: {dataset_train}")

print("\n" + "=" * 50)

# Kiểm tra thư mục validation
if os.path.exists(dataset_val):
    print(f"✅ Validation dataset found: {dataset_val}")
    
    val_people_count = 0
    val_total_images = 0
    
    for person_name in os.listdir(dataset_val):
        person_dir = os.path.join(dataset_val, person_name)
        if os.path.isdir(person_dir):
            val_people_count += 1
            images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            val_total_images += image_count
            print(f"   👤 {person_name}: {image_count} images")
    
    print(f"\n📊 Validation Summary:")
    print(f"   👥 Total people: {val_people_count}")
    print(f"   🖼️  Total images: {val_total_images}")
else:
    print(f"❌ Validation dataset not found: {dataset_val}")


# 

# In[1]:


import pickle
import numpy as np
import cv2
import ipywidgets as widgets
from IPython.display import display, Image as IPImage, clear_output
import io
from numpy.linalg import norm

# Hàm load database (sẽ gọi khi cần reload)
def load_database():
    db_path = 'models/face_database_kaggle.pkl'
    with open(db_path, 'rb') as f:
        return pickle.load(f)

database = load_database()  # Load ban đầu
print(f"✅ Loaded database with {len(database)} people.")

# Widget cho phần thêm người mới
name_input = widgets.Text(value='', placeholder='Nhập tên người mới (ví dụ: john_doe)', description='Tên:')
uploader_add = widgets.FileUpload(accept='image/*', multiple=True)  # Cho phép upload nhiều ảnh
add_button = widgets.Button(description='Thêm vào DB')
add_output = widgets.Output()

def on_add_button_clicked(b):
    global database  # Để cập nhật global database
    with add_output:
        clear_output()
        new_name = name_input.value.strip()
        if not new_name:
            print("⚠️ Vui lòng nhập tên!")
            return
        if new_name in database:
            print(f"⚠️ Tên '{new_name}' đã tồn tại trong DB!")
            return
        if not uploader_add.value:
            print("⚠️ Vui lòng upload ít nhất 1 ảnh!")
            return
        
        embeddings = []
        for file_info in uploader_add.value:
            image_bytes = file_info['content']
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            faces = model.get(img)
            if len(faces) == 0:
                print(f"⚠️ Ảnh '{file_info['name']}': Không phát hiện khuôn mặt.")
                continue
            embeddings.append(faces[0].embedding)
            print(f"✅ Ảnh '{file_info['name']}': Đã trích xuất embedding.")
        
        if not embeddings:
            print("⚠️ Không có embedding hợp lệ!")
            return
        
        mean_emb = np.mean(embeddings, axis=0)
        database[new_name] = mean_emb
        with open(db_path, 'wb') as f:
            pickle.dump(database, f)
        print(f"🎉 Đã thêm '{new_name}' vào DB với {len(embeddings)} ảnh. Tổng người: {len(database)}")
        
        # Reset widget
        name_input.value = ''
        uploader_add.value = ()

add_button.on_click(on_add_button_clicked)

# Hiển thị widget thêm người
print("\n➕ Thêm người mới vào DB:")
display(name_input, uploader_add, add_button, add_output)

# Widget cho phần test
uploader_test = widgets.FileUpload(accept='image/*', multiple=False)
test_button = widgets.Button(description='Test Ảnh')
test_output = widgets.Output()

def predict_face(embedding, threshold=0.5):
    query = embedding / norm(embedding)
    best_score = -1
    best_name = "Unknown"
    for name, emb in database.items():
        e = emb / norm(emb)
        score = float(np.dot(query, e))
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score

def on_test_button_clicked(b):
    global database  # Reload để chắc chắn dùng DB mới nhất
    database = load_database()  # Reload DB trước khi test
    with test_output:
        clear_output()
        if not uploader_test.value:
            print("⚠️ Vui lòng upload ảnh test!")
            return
        file_info = uploader_test.value[0]
        image_bytes = file_info['content']
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = model.get(img)
        if len(faces) == 0:
            print("⚠️ Không phát hiện khuôn mặt trong ảnh test.")
            return
        query_emb = faces[0].embedding
        pred_name, score = predict_face(query_emb)
        display(IPImage(data=image_bytes, width=300))
        print(f"🎯 Predicted: {pred_name} (Score: {score:.4f})")
        
        # Reset uploader
        uploader_test.value = ()

test_button.on_click(on_test_button_clicked)

# Hiển thị widget test
print("\n🧪 Test ảnh mới:")
display(uploader_test, test_button, test_output)

