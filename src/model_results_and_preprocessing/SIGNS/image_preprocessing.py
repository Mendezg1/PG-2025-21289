import os
from PIL import Image, ImageEnhance, ImageOps
import random
import shutil

STOP_IMAGES_DIR = 'src/SIGNS/alto'
STOP_LABELS_DIR = 'src/SIGNS/alto_labels'
UTURN_IMAGES_DIR = 'src/SIGNS/girou'
UTURN_LABELS_DIR = 'src/SIGNS/girou_labels'
YOLO_IMAGES_DIR = 'src/SIGNS/yolo'


# Function to rename image files in a directory to a sequential format
def rename_image_files(directory):
    
    counter = 1
    for filename in os.listdir(directory):
        new_name = f"{counter}.png"
        old_path = os.path.join(directory, filename)
        
        image = Image.open(old_path)
        image.save(os.path.join(directory, new_name))

        os.remove(old_path)

        print(f'Renamed: {filename} to {new_name}')
        counter += 1

def merge_classes(dir1, dir2):

    merge = []
    images1 = os.listdir(dir1)
    images2 = os.listdir(dir2)
    counter = 1

    for img1 in images1:
        img1_path = os.path.join(dir1, img1)
        image1 = Image.open(img1_path).convert("RGB")
        new_name = f"{counter}.png"
        merge.append((image1, new_name))
        counter += 1

    for img2 in images2:
        img2_path = os.path.join(dir2, img2)
        image2 = Image.open(img2_path).convert("RGB")
        new_name = f"{counter}.png"
        merge.append((image2, new_name))
        counter += 1

    return merge

# Makes a train-test split of images in a directory
def train_split(directories, train_ratio=0.8):
    images = merge_classes(directories[0], directories[1])

    random.shuffle(images)
    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_dir = os.path.join(YOLO_IMAGES_DIR, 'train')
    test_dir = os.path.join(YOLO_IMAGES_DIR, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for img, name in train_images:
        new_path = os.path.join(train_dir, name)
        img.save(new_path)

    for img, name in test_images:
        new_path = os.path.join(test_dir, name)
        img.save(new_path)


# Function to apply random transformations to an image
def apply_random_transform(image):
    angle = random.uniform(-30, 30)
    image = image.rotate(angle)

    scale_factor = random.uniform(0.9, 1.1)
    new_size = tuple([int(dim * scale_factor) for dim in image.size])
    image = image.resize(new_size, Image.BICUBIC)

    image = ImageOps.fit(image, image.size, method=Image.BICUBIC)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    return image

# Function to augment images in a directory, specifically for the lower class
def data_augmentation(dir1, dir2):
    higher_class = os.listdir(dir1)
    lower_class = os.listdir(dir2)
    higher_class_count = len(higher_class)
    lower_class_count = len(lower_class)

    if higher_class_count > lower_class_count:
        augmentation_factor = higher_class_count // lower_class_count
        current_count = len(lower_class)

        for _ in range(augmentation_factor - 1):
            for img in lower_class:
                img_path = os.path.join(dir2, img)

                image = Image.open(img_path).convert("RGB")

                augmented_image = apply_random_transform(image)

                new_img_path = os.path.join(dir2, f"{current_count}.png")
                augmented_image.save(new_img_path)
                current_count += 1
    

# Rename stop images
#rename_image_files(STOP_IMAGES_DIR)
#rename_image_files(UTURN_IMAGES_DIR)

# Perform data augmentation for uturn signs
#data_augmentation(dir1=STOP_IMAGES_DIR, dir2=UTURN_IMAGES_DIR)

# Train-test split for stop images
train_split([STOP_IMAGES_DIR, UTURN_IMAGES_DIR])

