import os
import shutil
import pandas as pd

# Define paths
base_dir = '/root/documentclassification/JL/datasets'
train_dir = os.path.join(base_dir, 'train')
train_csv = os.path.join(base_dir, 'train.csv')
meta_csv = os.path.join(base_dir, 'meta.csv')

# Read the CSV files
train_df = pd.read_csv(train_csv)
meta_df = pd.read_csv(meta_csv)

# Create a mapping from class number to class name
class_mapping = dict(zip(meta_df['target'], meta_df['class_name']))

# Create output directory
output_dir = os.path.join(base_dir, 'organized_train')
os.makedirs(output_dir, exist_ok=True)

# Create class folders
for class_name in class_mapping.values():
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    print(f"Created directory: {class_dir}")

# Copy images to their respective class folders
for _, row in train_df.iterrows():
    img_name = row['ID']
    class_num = row['target']
    class_name = class_mapping[class_num]
    
    src = os.path.join(train_dir, img_name)
    dst = os.path.join(output_dir, class_name, img_name)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {img_name} to {class_name}/")
    else:
        print(f"Warning: Source file not found: {src}")

print("\nOrganization complete!")
print(f"Images have been organized into class folders in: {output_dir}")
