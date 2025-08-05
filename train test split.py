import os
import random
import shutil

split_ratio = 0.8  # 80% training, 20% val

image_dir = 'images'
label_dir = 'labels'


train_img_dir = os.path.join(image_dir, 'train')
val_img_dir = os.path.join(image_dir, 'val')
train_lbl_dir = os.path.join(label_dir, 'train')
val_lbl_dir = os.path.join(label_dir, 'val')

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and os.path.isfile(os.path.join(image_dir, f))]

random.shuffle(image_files)
split_idx = int(len(image_files) * split_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(file_list, img_dest, lbl_dest):
    for fname in file_list:
        base = os.path.splitext(fname)[0]
        img_src = os.path.join(image_dir, fname)
        lbl_src = os.path.join(label_dir, base + '.txt')

        
        if os.path.exists(img_src):
            shutil.move(img_src, os.path.join(img_dest, fname))
        
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(lbl_dest, base + '.txt'))

move_files(train_files, train_img_dir, train_lbl_dir)
move_files(val_files, val_img_dir, val_lbl_dir)

print(f"Sukses! Total: {len(image_files)} gambar → {len(train_files)} train, {len(val_files)} val.")
