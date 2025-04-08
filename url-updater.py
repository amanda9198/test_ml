import os
import re
import sys

def update_label_files(url_file, label_dir):
    """
    Update label filenames to match the image URLs with random characters.
    
    :param url_file: Path to the file containing image URLs
    :param label_dir: Directory containing label files
    """
    # Get absolute paths
    url_file = os.path.abspath(url_file)
    label_dir = os.path.abspath(label_dir)
    
    print(f"\nWorking with:")
    print(f"URL File: {url_file}")
    print(f"Label Directory: {label_dir}")
    
    # Verify files and directories exist
    if not os.path.exists(url_file):
        print(f"ERROR: URL file not found: {url_file}")
        return
    if not os.path.exists(label_dir):
        print(f"ERROR: Label directory not found: {label_dir}")
        return
    
    # Read URLs
    try:
        with open(url_file, 'r') as f:
            urls = f.read().splitlines()
    except Exception as e:
        print(f"Error reading URL file: {e}")
        return
    
    # Create a mapping of image numbers to their full URL with random characters
    url_map = {}
    for url in urls:
        # Extract image number and random characters from URL
        match = re.search(r'image-(\d+)_([A-Za-z0-9]+)\.jpg', url)
        if match:
            image_num = match.group(1)
            random_chars = match.group(2)
            url_map[image_num] = random_chars
    
    # Debug: print URL map size
    print(f"\nTotal URLs processed: {len(urls)}")
    print(f"Unique image numbers in URL map: {len(url_map)}")
    
    # List current label files
    print("\nCurrent label files:")
    current_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(current_labels)
    
    # Process label files
    renamed_count = 0
    for label_file in current_labels:
        # Extract image number from label filename
        match = re.match(r'image-(\d+)\.txt', label_file)
        if not match:
            print(f"Skipping non-standard label file: {label_file}")
            continue
        
        image_num = match.group(1)
        
        # Check if we have a matching URL with random characters
        if image_num not in url_map:
            print(f"No matching URL found for label {label_file}")
            continue
        
        # Path to original and new label files
        original_label_path = os.path.join(label_dir, label_file)
        new_filename = f"image-{image_num}_{url_map[image_num]}.txt"
        new_label_path = os.path.join(label_dir, new_filename)
        
        # Rename the label file
        try:
            os.rename(original_label_path, new_label_path)
            print(f"Renamed {label_file} to {new_filename}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {label_file}: {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")

def main():
    # Paths to your files (adjust as needed)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Train dataset
    train_url_file = os.path.join(current_dir, 'datasets', 'url_dataset', 'train.txt')
    train_label_dir = os.path.join(current_dir, 'datasets', 'url_dataset', 'labels', 'train')
    
    # Validation dataset
    val_url_file = os.path.join(current_dir, 'datasets', 'url_dataset', 'val.txt')
    val_label_dir = os.path.join(current_dir, 'datasets', 'url_dataset', 'labels', 'val')
    
    # Update labels for train URLs
    print("\n--- Updating labels for train URLs ---")
    update_label_files(train_url_file, train_label_dir)
    
    # Update labels for validation URLs
    print("\n--- Updating labels for validation URLs ---")
    update_label_files(val_url_file, val_label_dir)

if __name__ == "__main__":
    main()