import os
import glob

# Specify your image directory
image_dir = r"C:\Users\joyal\Desktop\to become machine learning enginerr\Generative Ai\Image Generation with GANs\img_celeba"

# Get all jpg image paths in the directory
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

# Check if images are found
if image_paths:
    print(f"Found {len(image_paths)} images in the folder:")
    for path in image_paths:
        print(path)  # Print each image path
else:
    print("No images found in the specified folder.")
