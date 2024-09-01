from PIL import Image, ImageDraw, ImageFont
import os

# Path to the directory containing images and coordinate files
image_dir = "/home/ubuntu/yolo-webapp/RawImage/TrainingData"
coord_dir = "/home/ubuntu/yolo-webapp/AnnotationsByMD/400_junior"
output_dir = "/home/ubuntu/yolo-webapp/jtest"

# Make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Font settings for the labels
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Change as per your system
font_size = 12
font = ImageFont.truetype(font_path, font_size)

# Iterate over all image files
for image_filename in os.listdir(image_dir):
    if image_filename.endswith(('.bmp', '.jpg', '.jpeg')):
        print(image_filename)
        image_path = os.path.join(image_dir, image_filename)
        coord_path = os.path.join(coord_dir, os.path.splitext(image_filename)[0] + '.txt')
        output_path = os.path.join(output_dir, image_filename)

        # Open the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Read the coordinates from the text file
        print(coord_path)
        with open(coord_path, 'r') as f:
            for label, line in enumerate(f):
                if label >=19:
                    break
                parts = line.strip().split(',')
                #label = parts[0]
                x, y = int(parts[0]), int(parts[1])
                # Draw the point and label on the image
                draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
                #draw.text((x+5, y-5), label, fill='red', font=font)

        # Save the image with the labels
        image.save(output_path)
        print(f'Saved: {output_path}')

print('Processing complete.')
from PIL import Image, ImageDraw, ImageFont
import os

# Path to the directory containing images and coordinate files
image_dir = "/home/ubuntu/yolo-webapp/RawImage/TrainingData"
coord_dir = "/home/ubuntu/yolo-webapp/AnnotationsByMD/400_junior"
output_dir = "/home/ubuntu/yolo-webapp/jtest"

# Make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Font settings for the labels
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Change as per your system
font_size = 12
font = ImageFont.truetype(font_path, font_size)

# Iterate over all image files
for image_filename in os.listdir(image_dir):
    if image_filename.endswith(('.bmp', '.jpg', '.jpeg')):
        print(image_filename)
        image_path = os.path.join(image_dir, image_filename)
        coord_path = os.path.join(coord_dir, os.path.splitext(image_filename)[0] + '.txt')
        output_path = os.path.join(output_dir, image_filename)

        # Open the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Read the coordinates from the text file
        print(coord_path)
        with open(coord_path, 'r') as f:
            for label, line in enumerate(f):
                if label >=19:
                    break
                parts = line.strip().split(',')
                #label = parts[0]
                x, y = int(parts[0]), int(parts[1])
                # Draw the point and label on the image
                draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
                #draw.text((x+5, y-5), label, fill='red', font=font)

        # Save the image with the labels
        image.save(output_path)
        print(f'Saved: {output_path}')

print('Processing complete.')
