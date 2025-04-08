import os
import yaml
import shutil
import argparse
from pathlib import Path
import urllib.request

def download_images_for_testing(url_file, output_dir, max_images=10):
    """
    Download a few images for testing purposes
    
    Args:
        url_file: Path to the file containing image URLs (train.txt or val.txt)
        output_dir: Directory to save downloaded images
        max_images: Maximum number of images to download
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(url_file, 'r') as f:
        urls = f.read().splitlines()
    
    # Limit to max_images
    urls = urls[:max_images]
    
    for url in urls:
        try:
            # Extract filename from URL
            filename = os.path.basename(url)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Downloading {url} to {output_path}")
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def create_local_dataset_yaml(url_dataset_dir, output_dir):
    """
    Create a version of dataset.yaml for local use
    
    Args:
        url_dataset_dir: Directory containing the URL-based dataset
        output_dir: Directory for the local dataset
    """
    # First, copy the labels directory
    shutil.copytree(
        os.path.join(url_dataset_dir, 'labels'),
        os.path.join(output_dir, 'labels'),
        dirs_exist_ok=True
    )
    
    # Create directories for images
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    
    # Load the original dataset.yaml
    with open(os.path.join(url_dataset_dir, 'dataset.yaml'), 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Update paths for local use
    dataset_config['path'] = os.path.abspath(output_dir)
    dataset_config['train'] = 'images/train'
    dataset_config['val'] = 'images/val'
    
    # Save the new dataset.yaml
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created local dataset.yaml at {os.path.join(output_dir, 'dataset.yaml')}")
    print("You'll need to manually download images and place them in images/train and images/val directories.")

def create_custom_model_config(output_dir, num_classes=2):
    """
    Create a custom YOLOv5 model config file
    
    Args:
        output_dir: Directory to save the model config
        num_classes: Number of classes in the dataset
    """
    # YOLOv5s config with modified number of classes
    config_content = f"""
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
"""
    
    config_path = os.path.join(output_dir, 'yolov5s_custom.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
        
    print(f"Created custom model config at {config_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset Utilities')
    parser.add_argument('--command', choices=['download', 'local_yaml', 'model_config'], required=True,
                        help='Command to execute')
    parser.add_argument('--url-dataset', help='Directory containing URL-based dataset')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--url-file', help='Path to URL file (train.txt or val.txt)')
    parser.add_argument('--max-images', type=int, default=10, help='Maximum images to download')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes for model config')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        if not args.url_file:
            parser.error("--url-file is required for download command")
        download_images_for_testing(args.url_file, args.output_dir, args.max_images)
    elif args.command == 'local_yaml':
        if not args.url_dataset:
            parser.error("--url-dataset is required for local_yaml command")
        create_local_dataset_yaml(args.url_dataset, args.output_dir)
    elif args.command == 'model_config':
        create_custom_model_config(args.output_dir, args.num_classes)

if __name__ == "__main__":
    main()