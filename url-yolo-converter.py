import os
import yaml
import re
import random
import argparse
from pathlib import Path
import urllib.request
import urllib.error
import shutil
from tqdm import tqdm
import concurrent.futures

def get_actual_image_url(base_url, image_number):
    """
    Find the actual image URL by checking the directory listing.
    
    Args:
        base_url: The base URL for the imageset
        image_number: The image number (e.g., '0000601')
    
    Returns:
        Complete URL if found, None otherwise
    """
    try:
        # First, try to list the directory to find the actual filename
        with urllib.request.urlopen(base_url) as response:
            html = response.read().decode('utf-8')
            
            # Look for image with the matching number
            pattern = f'image-{image_number}_[a-zA-Z0-9]+\\.jpg'
            matches = re.findall(pattern, html)
            
            if matches:
                return f"{base_url.rstrip('/')}/{matches[0]}"
    except (urllib.error.URLError, urllib.error.HTTPError):
        pass
        
    # If we can't find it, try some common extensions
    for ext in ['jpg', 'jpeg', 'png']:
        test_url = f"{base_url.rstrip('/')}/image-{image_number}.{ext}"
        try:
            with urllib.request.urlopen(test_url) as response:
                if response.getcode() == 200:
                    return test_url
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
    
    # Fallback to a best guess
    return f"{base_url.rstrip('/')}/image-{image_number}.jpg"

def parse_annotation_format(annotation_line):
    """Parse annotation line in the format '0, 1004, 360, 1020, 374, 0'"""
    values = [int(x.strip()) for x in annotation_line.split(',')]
    if len(values) >= 6:
        class_id = values[0]  # This should be remapped to your classes
        x1, y1, x2, y2 = values[1:5]
        return {
            'class_id': class_id,
            'x1': x1,
            'y1': y1, 
            'x2': x2,
            'y2': y2
        }
    return None

def convert_to_yolo_format(annotation, img_width, img_height, class_mapping=None):
    """
    Convert annotation to YOLO format.
    
    Args:
        annotation: Parsed annotation dict
        img_width, img_height: Image dimensions
        class_mapping: Optional mapping from original class IDs to YOLO class IDs
    
    Returns:
        String in YOLO format: "class_id x_center y_center width height"
    """
    class_id = annotation['class_id']
    if class_mapping and class_id in class_mapping:
        class_id = class_mapping[class_id]
        
    x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
    
    # Calculate normalized center coordinates, width and height
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_yaml_to_yolo(yaml_content, base_url, output_dir, class_mapping=None, verify_urls=False):
    """
    Process YAML annotation file to create YOLO dataset with URLs
    
    Args:
        yaml_content: String content of the YAML file
        base_url: Base URL for the imageset
        output_dir: Directory to save YOLO label files
        class_mapping: Mapping from original class IDs to YOLO class IDs
        verify_urls: Whether to verify each URL exists
    
    Returns:
        List of (image_url, label_path) pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = yaml.safe_load(yaml_content)
    image_label_pairs = []
    
    for img_entry in tqdm(data.get('images', []), desc="Processing annotations"):
        meta = img_entry.get('meta', '')
        annotations = img_entry.get('annotations', [])
        
        # Extract image number and dimensions
        meta_match = re.search(r'image-(\d+)\.jpg,\s*(\d+),\s*(\d+)', meta)
        if not meta_match:
            continue
            
        image_number = meta_match.group(1)
        img_width = int(meta_match.group(2))
        img_height = int(meta_match.group(3))
        
        # Get image URL
        image_url = get_actual_image_url(base_url, image_number)
        
        if verify_urls:
            try:
                with urllib.request.urlopen(image_url) as response:
                    if response.getcode() != 200:
                        print(f"Warning: URL returned status {response.getcode()}: {image_url}")
                        continue
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"Warning: Failed to verify URL {image_url}: {e}")
                continue
        
        # Process annotations
        yolo_annotations = []
        for annotation_line in annotations:
            annotation = parse_annotation_format(annotation_line)
            if annotation:
                yolo_annotation = convert_to_yolo_format(annotation, img_width, img_height, class_mapping)
                yolo_annotations.append(yolo_annotation)
        
        if yolo_annotations:
            # Create label file
            label_filename = f"image-{image_number}.txt"
            label_path = os.path.join(output_dir, label_filename)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
                
            image_label_pairs.append((image_url, label_path))
    
    return image_label_pairs

def create_url_dataset(imagesets, annotation_files, output_dir, base_url_template, 
                       class_mapping=None, split_ratio=0.8, verify_urls=False):
    """
    Create a YOLO dataset using URLs
    
    Args:
        imagesets: List of imageset IDs (e.g., ['145', '146'])
        annotation_files: Dictionary mapping imageset IDs to annotation file paths
        output_dir: Directory to save dataset
        base_url_template: Template for base URL (e.g., 'https://prism-static.aruw.org/images/1_{id}/')
        class_mapping: Mapping from original class IDs to YOLO class IDs
        split_ratio: Train/val split ratio
        verify_urls: Whether to verify each URL exists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for labels
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Lists to store train and val data
    train_data = []
    val_data = []
    
    for imageset_id in imagesets:
        if imageset_id not in annotation_files:
            print(f"Warning: No annotation file for imageset {imageset_id}")
            continue
            
        annotation_file = annotation_files[imageset_id]
        base_url = base_url_template.format(id=imageset_id)
        
        print(f"Processing imageset {imageset_id} with annotation {annotation_file}")
        try:
            with open(annotation_file, 'r') as f:
                yaml_content = f.read()
        except Exception as e:
            print(f"Error reading annotation file: {e}")
            continue
            
        # Create temporary label directory for this imageset
        temp_labels_dir = os.path.join(output_dir, 'temp_labels', imageset_id)
        os.makedirs(temp_labels_dir, exist_ok=True)
        
        # Process annotations
        image_label_pairs = process_yaml_to_yolo(yaml_content, base_url, temp_labels_dir, 
                                                class_mapping, verify_urls)
        
        # Split into train/val
        random.shuffle(image_label_pairs)
        split_idx = int(len(image_label_pairs) * split_ratio)
        train_pairs = image_label_pairs[:split_idx]
        val_pairs = image_label_pairs[split_idx:]
        
        # Copy labels to train/val directories
        for image_url, label_path in train_pairs:
            label_filename = os.path.basename(label_path)
            new_label_path = os.path.join(train_labels_dir, label_filename)
            shutil.copy(label_path, new_label_path)
            train_data.append((image_url, new_label_path))
            
        for image_url, label_path in val_pairs:
            label_filename = os.path.basename(label_path)
            new_label_path = os.path.join(val_labels_dir, label_filename)
            shutil.copy(label_path, new_label_path)
            val_data.append((image_url, new_label_path))
            
    # Create train.txt and val.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for image_url, _ in train_data:
            f.write(f"{image_url}\n")
            
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for image_url, _ in val_data:
            f.write(f"{image_url}\n")
            
    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(output_dir),  # Absolute path
        'train': 'train.txt',  # Path to train.txt
        'val': 'val.txt',      # Path to val.txt
        'test': '',           # No test set
        'nc': len(class_mapping) if class_mapping else 2,  # Number of classes
        'names': {i: name for i, name in enumerate(["red", "blue"])}  # Class names
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
        
    print(f"Dataset created with {len(train_data)} training and {len(val_data)} validation images")
    print(f"Dataset YAML saved to {os.path.join(output_dir, 'dataset.yaml')}")
    
    # Clean up temporary files
    shutil.rmtree(os.path.join(output_dir, 'temp_labels'), ignore_errors=True)

def find_annotation_files(annotations_dir, imagesets):
    """
    Find annotation files for given imagesets
    
    Args:
        annotations_dir: Directory containing annotation files
        imagesets: List of imageset IDs (e.g., ['145', '146'])
        
    Returns:
        Dictionary mapping imageset IDs to annotation file paths
    """
    annotation_files = {}
    
    for imageset_id in imagesets:
        # Try different patterns to find annotation file
        patterns = [
            f"*__blue_{imageset_id}__*.yaml",
            f"*__blue_{imageset_id}.yaml",
            f"*_blue_{imageset_id}.yaml",
            f"*_{imageset_id}_*.yaml",
            f"*_{imageset_id}.yaml"
        ]
        
        found = False
        for pattern in patterns:
            matching_files = list(Path(annotations_dir).glob(pattern))
            if matching_files:
                annotation_files[imageset_id] = str(matching_files[0])
                found = True
                break
                
        if not found:
            # Content-based matching as fallback
            print(f"No annotation file pattern match for imageset {imageset_id}, trying content-based matching...")
            for yaml_file in Path(annotations_dir).glob("*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        content = f.read()
                        # Check if this yaml contains images from this imageset
                        if re.search(rf'blue_{imageset_id}|_{imageset_id}_|_{imageset_id}/', content):
                            annotation_files[imageset_id] = str(yaml_file)
                            print(f"  Found match: {yaml_file.name}")
                            found = True
                            break
                except Exception:
                    pass
    
    return annotation_files

def main():
    parser = argparse.ArgumentParser(description='Create YOLO dataset using URLs')
    parser.add_argument('--imagesets', nargs='+', required=True, 
                        help='List of imageset IDs (e.g., 145 146)')
    parser.add_argument('--annotations-dir', required=True,
                        help='Directory containing annotation YAML files')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save dataset')
    parser.add_argument('--url-template', default='https://prism-static.aruw.org/images/1_{id}/',
                        help='Template for image URLs')
    parser.add_argument('--verify-urls', action='store_true',
                        help='Verify URLs exist (slower)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    
    args = parser.parse_args()
    
    # Find annotation files for imagesets
    annotation_files = find_annotation_files(args.annotations_dir, args.imagesets)
    
    # Class mapping (adjust based on your needs)
    # From the YAML, looks like:
    # 1 -> blue (class_id 1)
    # 3 -> also blue? (needs clarification)
    # 10 -> red (class_id 0)
    class_mapping = {
        1: 1,  # Blue
        3: 1,  # Also blue (based on supported_label_types)
        10: 0  # Red
    }
    
    create_url_dataset(args.imagesets, annotation_files, args.output_dir,
                      args.url_template, class_mapping, args.split_ratio,
                      args.verify_urls)

if __name__ == "__main__":
    main()