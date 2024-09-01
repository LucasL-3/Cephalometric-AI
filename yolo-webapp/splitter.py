import glob
import os
import shutil
import random

# Paths to the source and destination directories
source_dir = '/home/ubuntu/yolo-webapp/cephdata/images/all'
train_dir = '/home/ubuntu/yolo-webapp/cephdata/images/train'
test_dir = '/home/ubuntu/yolo-webapp/cephdata/images/test'
val_dir = '/home/ubuntu/yolo-webapp/cephdata/images/val'

# Ensure the destination directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of all files in the source directory
files = glob.glob(os.path.join(source_dir, '*'))

# Shuffle the list of files
random.shuffle(files)

# Calculate the number of files for each set
total_files = len(files)
train_count = int(0.8 * total_files)
test_count = int(0.15 * total_files)
val_count = total_files - train_count - test_count  # Ensure all files are used

# Split the files into training, test, and validation sets
train_files = files[:train_count]
test_files = files[train_count:train_count + test_count]
val_files = files[train_count + test_count:]

# Function to copy files to a destination directory
def copy_files(files, dest_dir):
    for file in files:
        shutil.copy(file, dest_dir)

# Copy the files to their respective directories
copy_files(train_files, train_dir)
copy_files(test_files, test_dir)
copy_files(val_files, val_dir)

print(f'Total files: {total_files}')
print(f'Training files: {len(train_files)}')
print(f'Test files: {len(test_files)}')
print(f'Validation files: {len(val_files)}')
import glob
import os
import shutil
import random

# Paths to the source and destination directories
source_dir = '/home/ubuntu/yolo-webapp/cephdata/images/all'
train_dir = '/home/ubuntu/yolo-webapp/cephdata/images/train'
test_dir = '/home/ubuntu/yolo-webapp/cephdata/images/test'
val_dir = '/home/ubuntu/yolo-webapp/cephdata/images/val'

# Ensure the destination directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of all files in the source directory
files = glob.glob(os.path.join(source_dir, '*'))

# Shuffle the list of files
random.shuffle(files)

# Calculate the number of files for each set
total_files = len(files)
train_count = int(0.8 * total_files)
test_count = int(0.15 * total_files)
val_count = total_files - train_count - test_count  # Ensure all files are used

# Split the files into training, test, and validation sets
train_files = files[:train_count]
test_files = files[train_count:train_count + test_count]
val_files = files[train_count + test_count:]

# Function to copy files to a destination directory
def copy_files(files, dest_dir):
    for file in files:
        shutil.copy(file, dest_dir)

# Copy the files to their respective directories
copy_files(train_files, train_dir)
copy_files(test_files, test_dir)
copy_files(val_files, val_dir)

print(f'Total files: {total_files}')
print(f'Training files: {len(train_files)}')
print(f'Test files: {len(test_files)}')
print(f'Validation files: {len(val_files)}')
