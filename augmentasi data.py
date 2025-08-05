import os
import cv2
import albumentations as A
kelas_target = '2'
jumlah_aug_per_gambar = 5
dominasi_minimal = 0.5
input_image_dir = 'images'
input_label_dir = 'labels'
output_image_dir = 'augmented/images'
output_label_dir = 'augmented/labels'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
transform = A.Compose([
 A.HorizontalFlip(p=0.5),
 A.RandomBrightnessContrast(p=0.5),
 A.Rotate(limit=20, p=0.5),
 A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
 A.RandomGamma(p=0.3),
 A.Blur(blur_limit=3, p=0.2),], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
for filename in os.listdir(input_label_dir):
 if not filename.endswith('.txt'):
 continue
 label_path = os.path.join(input_label_dir, filename)
 image_name = filename.replace('.txt', '.jpg') 
 image_path = os.path.join(input_image_dir, image_name)
 if not os.path.exists(image_path):
 continue
 with open(label_path, 'r') as f:
 label_lines = [line.strip() for line in f.readlines() if line.strip()]
 if not label_lines:
 continue
 class_ids = [line.split()[0] for line in label_lines]
 total = len(class_ids)
 jumlah_kelas2 = class_ids.count(kelas_target)
 if jumlah_kelas2 / total < dominasi_minimal:
 continue # skip jika kelas 2 tidak dominan
 image = cv2.imread(image_path)
 h, w = image.shape[:2]
 boxes = []
 class_labels = []
 for line in label_lines:
 parts = line.split()
 if len(parts) == 5:
 class_id = int(parts[0])
 bbox = list(map(float, parts[1:]))
 boxes.append(bbox)
 class_labels.append(class_id)
 for i in range(jumlah_aug_per_gambar):
 augmented = transform(image=image, bboxes=boxes,
class_labels=class_labels)
 aug_img = augmented['image']
 aug_bboxes = []
 aug_labels = []
 for cid, bbox in zip(augmented['class_labels'], augmented['bboxes']):
 x, y, w_, h_ = bbox
 if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w_ <= 1 and 0 <= h_ <= 1:
 if w_ > 0 and h_ > 0:
 aug_bboxes.append([x, y, w_, h_])
 aug_labels.append(cid)
 if not aug_bboxes:
 continue
 new_image_name = image_name.replace('.jpg', f'_aug{i}.jpg')
 new_label_name = new_image_name.replace('.jpg', '.txt')
 cv2.imwrite(os.path.join(output_image_dir, new_image_name), aug_img)
 with open(os.path.join(output_label_dir, new_label_name), 'w') as f:
 for cid, bbox in zip(aug_labels, aug_bboxes):
 f.write(f"{int(cid)} {' '.join(map(str, bbox))}\n")
print("✅ Augmentasi selesai untuk gambar dengan dominasi kelas 2.")
