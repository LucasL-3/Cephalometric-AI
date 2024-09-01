import os
import glob
from PIL import Image

def remove_last_8_lines_and_normalize(input_folder, output_folder, image_folder):

    files = sorted(glob.glob(input_folder + "/*.txt"))
    
    # Print the list of files found
    for file in files:
        print(file)

    for filename in files:
        filename=os.path.basename(filename)
        print(filename)
        if filename.endswith('.txt'):
            input_text_file = os.path.join(input_folder, filename)
            output_text_file = os.path.join(output_folder, filename)
            image_file = os.path.join(image_folder, filename.replace('.txt', '.bmp'))
            if not os.path.exists(image_file):
               continue
            # Read the image to get its dimensions
            image = Image.open(image_file)
            width, height = image.size

            # Read the text file
            with open(input_text_file, 'r') as file:
                lines = file.readlines()
            
            # Remove the last 8 lines
            lines = lines[:-8]

            # Start with the two initial coordinates
            normalized_lines = ["0.5,0.5", "1.0,1.0"]
        
            # Normalize the remaining coordinates
            for line in lines:
                x, y = map(int, line.strip().split(','))
                normalized_x = x / width
                normalized_y = y / height
                normalized_lines.append(f"{normalized_x:.4f},{normalized_y:.4f}")

            # Combine all coordinates into a single line separated by spaces
            combined_line = " ".join(normalized_lines)

            # Write the combined line to the output file
            with open(output_text_file, 'w') as file:
                file.write(combined_line + '\n')

# Example usage
remove_last_8_lines_and_normalize('/home/ubuntu/yolo-webapp/AnnotationsByMD/400_junior/', '/home/ubuntu/yolo-webapp/cephdata/labels/all/', '/home/ubuntu/yolo-webapp/RawImage/Test2Data/')import os
import glob
from PIL import Image

def remove_last_8_lines_and_normalize(input_folder, output_folder, image_folder):

    files = sorted(glob.glob(input_folder + "/*.txt"))
    
    # Print the list of files found
    for file in files:
        print(file)

    for filename in files:
        filename=os.path.basename(filename)
        print(filename)
        if filename.endswith('.txt'):
            input_text_file = os.path.join(input_folder, filename)
            output_text_file = os.path.join(output_folder, filename)
            image_file = os.path.join(image_folder, filename.replace('.txt', '.bmp'))
            if not os.path.exists(image_file):
               continue
            # Read the image to get its dimensions
            image = Image.open(image_file)
            width, height = image.size

            # Read the text file
            with open(input_text_file, 'r') as file:
                lines = file.readlines()
            
            # Remove the last 8 lines
            lines = lines[:-8]

            # Start with the two initial coordinates
            normalized_lines = ["0.5,0.5", "1.0,1.0"]
        
            # Normalize the remaining coordinates
            for line in lines:
                x, y = map(int, line.strip().split(','))
                normalized_x = x / width
                normalized_y = y / height
                normalized_lines.append(f"{normalized_x:.4f},{normalized_y:.4f}")

            # Combine all coordinates into a single line separated by spaces
            combined_line = " ".join(normalized_lines)

            # Write the combined line to the output file
            with open(output_text_file, 'w') as file:
                file.write(combined_line + '\n')

# Example usage
remove_last_8_lines_and_normalize('/home/ubuntu/yolo-webapp/AnnotationsByMD/400_junior/', '/home/ubuntu/yolo-webapp/cephdata/labels/all/', '/home/ubuntu/yolo-webapp/RawImage/Test2Data/')