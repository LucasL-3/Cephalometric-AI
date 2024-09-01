import glob
import os

# Paths to the label directories
label_dirs = [
    '/home/ubuntu/yolo-webapp/cephdata/labels/all',
    '/home/ubuntu/yolo-webapp/cephdata/labels/train',
    '/home/ubuntu/yolo-webapp/cephdata/labels/test',
    '/home/ubuntu/yolo-webapp/cephdata/labels/val'
]

# Function to clean the label files
def clean_label_files(label_dir):
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as file:
            content = file.read()
        cleaned_content = content.replace(',', ' ')
        with open(label_file, 'w') as file:
            file.write(cleaned_content)
        print(f'Cleaned label file: {label_file}')

# Process each directory
for label_dir in label_dirs:
    clean_label_files(label_dir)

print('All label files have been cleaned.')
import glob
import os

# Paths to the label directories
label_dirs = [
    '/home/ubuntu/yolo-webapp/cephdata/labels/all',
    '/home/ubuntu/yolo-webapp/cephdata/labels/train',
    '/home/ubuntu/yolo-webapp/cephdata/labels/test',
    '/home/ubuntu/yolo-webapp/cephdata/labels/val'
]

# Function to clean the label files
def clean_label_files(label_dir):
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as file:
            content = file.read()
        cleaned_content = content.replace(',', ' ')
        with open(label_file, 'w') as file:
            file.write(cleaned_content)
        print(f'Cleaned label file: {label_file}')

# Process each directory
for label_dir in label_dirs:
    clean_label_files(label_dir)

print('All label files have been cleaned.')
