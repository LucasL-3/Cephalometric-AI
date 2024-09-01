import os
import shutil
import glob

# Paths to the source and destination directories for labels
label_source_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/all'
label_train_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/train'
label_test_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/test'
label_val_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/val'

# Paths to the image directories
image_train_dir = '/home/ubuntu/yolo-webapp/cephdata/images/train'
image_test_dir = '/home/ubuntu/yolo-webapp/cephdata/images/test'
image_val_dir = '/home/ubuntu/yolo-webapp/cephdata/images/val'

# Ensure the destination directories exist
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_test_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Function to get the corresponding label file for a given image file
def get_label_file(image_file, source_label_dir):
    image_filename = os.path.basename(image_file)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    return os.path.join(source_label_dir, label_filename)

# Function to copy corresponding label files based on image files
def copy_label_files(image_dir, source_label_dir, dest_label_dir):
    image_files = glob.glob(os.path.join(image_dir, '*'))
    for image_file in image_files:
        label_file = get_label_file(image_file, source_label_dir)
        if os.path.exists(label_file):
            shutil.copy(label_file, dest_label_dir)

# Copy the corresponding label files to their respective directories
copy_label_files(image_train_dir, label_source_dir, label_train_dir)
copy_label_files(image_test_dir, label_source_dir, label_test_dir)
copy_label_files(image_val_dir, label_source_dir, label_val_dir)

print('Labels have been split according to the image splits.')
import os
import shutil
import glob

# Paths to the source and destination directories for labels
label_source_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/all'
label_train_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/train'
label_test_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/test'
label_val_dir = '/home/ubuntu/yolo-webapp/cephdata/labels/val'

# Paths to the image directories
image_train_dir = '/home/ubuntu/yolo-webapp/cephdata/images/train'
image_test_dir = '/home/ubuntu/yolo-webapp/cephdata/images/test'
image_val_dir = '/home/ubuntu/yolo-webapp/cephdata/images/val'

# Ensure the destination directories exist
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_test_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Function to get the corresponding label file for a given image file
def get_label_file(image_file, source_label_dir):
    image_filename = os.path.basename(image_file)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    return os.path.join(source_label_dir, label_filename)

# Function to copy corresponding label files based on image files
def copy_label_files(image_dir, source_label_dir, dest_label_dir):
    image_files = glob.glob(os.path.join(image_dir, '*'))
    for image_file in image_files:
        label_file = get_label_file(image_file, source_label_dir)
        if os.path.exists(label_file):
            shutil.copy(label_file, dest_label_dir)

# Copy the corresponding label files to their respective directories
copy_label_files(image_train_dir, label_source_dir, label_train_dir)
copy_label_files(image_test_dir, label_source_dir, label_test_dir)
copy_label_files(image_val_dir, label_source_dir, label_val_dir)

print('Labels have been split according to the image splits.')
