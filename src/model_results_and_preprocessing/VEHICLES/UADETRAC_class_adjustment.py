import os

labels_dirs = ["C:/Users/euroc/OneDrive/Escritorio/U/Grad/data/UADETRAC/labels/train", "C:/Users/euroc/OneDrive/Escritorio/U/Grad/data/UADETRAC/labels/val"]

def edit_classes():
    for dir_path in labels_dirs:
        for label_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, label_file)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                if class_id == 1:
                    new_class_id = 0
                elif class_id == 2:
                    new_class_id = 1
                elif class_id == 3:
                    new_class_id = 2
                new_lines.append(f"{new_class_id} " + " ".join(parts[1:]) + "\n")

            with open(file_path, "w") as f:
                f.writelines(new_lines)

def get_classes():
    set_classes = set()
    for dir_path in labels_dirs:
        for label_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, label_file)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.split()
                set_classes.add(int(parts[0]))

    print(set_classes)
get_classes()
