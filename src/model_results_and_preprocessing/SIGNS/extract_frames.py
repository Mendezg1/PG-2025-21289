import cv2
import os
import random
import shutil
from pathlib import Path

# video_dir = ["C:/Users/euroc/Pictures/grad/Grad/src/SIGNS/data_vids/alto", "C:/Users/euroc/Pictures/grad/Grad/src/SIGNS/data_vids/girou"]
# output_dir = "C:/Users/euroc/Pictures/grad/Grad/src/SIGNS/data/all"
# frame_skip = 9

# os.makedirs(output_dir, exist_ok=True)

# # Empieza en el último id de la data anotada
# count = 69

# # Procesar cada video en cada carpeta (alto, girou)
# for num in [0, 1]:
#     for video_file in Path(video_dir[num]).glob("*.mp4"):
#         cap = cv2.VideoCapture(str(video_file))
#         frame_id = 0
        
#         video_name = video_file.stem
#         video_output = os.path.join(output_dir, video_name)
#         os.makedirs(video_output, exist_ok=True)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Se guardan solo algunos frames para evitar redundancia
#             if frame_id % frame_skip == 0:
#                 frame_path = os.path.join(video_output, f"{count}.jpg")
#                 cv2.imwrite(frame_path, frame)
#                 count += 1

#             frame_id += 1

#         cap.release()
#         print(f"Extraídos {count} frames de {video_name}")

# print("Extracción completada")


base_dir = "C:/Users/euroc/Pictures/grad/Grad/src/SIGNS/data"
output_all = os.path.join(base_dir, "all")
output_train = os.path.join(base_dir, "train")
output_val = os.path.join(base_dir, "val")

all_images = list(Path(output_all).glob("*.jpg"))
random.shuffle(all_images)

# 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Mover archivos
for img_path in train_images:
    shutil.move(str(img_path), os.path.join(output_train, img_path.name))

for img_path in val_images:
    shutil.move(str(img_path), os.path.join(output_val, img_path.name))

print(f"Split completado: {len(train_images)} train / {len(val_images)} val")
