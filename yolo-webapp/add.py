import glob
import os

# Paths to the label directories
label_dirs = [
    '/home/ubuntu/yolo-webapp/cephdata/labels/all',
    '/home/ubuntu/yolo-webapp/cephdata/labels/train',
    '/home/ubuntu/yolo-webapp/cephdata/labels/test',
    '/home/ubuntu/yolo-webapp/cephdata/labels/val'
]

# Function to add '0' at the beginning of each line in label files
def add_integer_prefix(label_dir, integer_prefix='0'):
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as file:
            lines = file.readlines()
        
        # Add the integer prefix to each line
        modified_lines = [f'{integer_prefix} {line}' for line in lines]
        
        with open(label_file, 'w') as file:
            file.writelines(modified_lines)
        
        print(f'Updated label file: {label_file}')

# Process each directory
for label_dir in label_dirs:
    add_integer_prefix(label_dir)

print('All label files have been updated.')
import glob
import os

# Paths to the label directories
label_dirs = [
    '/home/ubuntu/yolo-webapp/cephdata/labels/all',
    '/home/ubuntu/yolo-webapp/cephdata/labels/train',
    '/home/ubuntu/yolo-webapp/cephdata/labels/test',
    '/home/ubuntu/yolo-webapp/cephdata/labels/val'
]

# Function to add '0' at the beginning of each line in label files
def add_integer_prefix(label_dir, integer_prefix='0'):
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as file:
            lines = file.readlines()
        
        # Add the integer prefix to each line
        modified_lines = [f'{integer_prefix} {line}' for line in lines]
        
        with open(label_file, 'w') as file:
            file.writelines(modified_lines)
        
        print(f'Updated label file: {label_file}')

# Process each directory
for label_dir in label_dirs:
    add_integer_prefix(label_dir)

print('All label files have been updated.')
